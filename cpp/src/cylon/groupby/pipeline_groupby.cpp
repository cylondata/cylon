/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_PIPELINE_CPP_
#define CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_PIPELINE_CPP_

#include <arrow/api.h>
#include <arrow/compute/api.h>

#include "../util/macros.hpp"
#include "../util/arrow_utils.hpp"
#include "../ctx/arrow_memory_pool_utils.hpp"

#include "pipeline_groupby.hpp"

namespace cylon {

typedef
arrow::Status (*AggregateFptr)(const arrow::Datum &array,
                               arrow::compute::ExecContext *fn_ctx,
                               arrow::Datum *res);

inline arrow::Status Sum(const arrow::Datum &array, arrow::compute::ExecContext *fn_ctx, arrow::Datum *res) {
  auto result = arrow::compute::Sum(array, fn_ctx);
  if (result.ok()) {
    *res = result.ValueOrDie();
  }
  return result.status();
}

inline arrow::Status Count(const arrow::Datum &array,
                           arrow::compute::ExecContext *fn_ctx,
                           arrow::Datum *res) {
  auto result = arrow::compute::Count(array, arrow::compute::CountOptions::Defaults(), fn_ctx);

  if (result.ok()) {
    *res = result.ValueOrDie();
  }
  return result.status();
}

template<bool minMax>
inline arrow::Status MinMax(const arrow::Datum &array,
                            arrow::compute::ExecContext *fn_ctx,
                            arrow::Datum *res) {
  auto result = arrow::compute::MinMax(array, arrow::compute::MinMaxOptions::Defaults(), fn_ctx);

  if (result.ok()) {
    arrow::Datum local_result = result.ValueOrDie(); // minmax returns a structscalar{min, max}
    const auto &struct_scalar = local_result.scalar_as<arrow::StructScalar>();
    *res = arrow::Datum(struct_scalar.value.at(minMax));
  }

  return result.status();
}

inline AggregateFptr PickAggregareFptr(const cylon::compute::AggregationOpId aggregation_op) {
  switch (aggregation_op) {
    case compute::SUM: return &Sum;
    case compute::COUNT:return &Count;
    case compute::MIN: return &MinMax<0>;
    case compute::MAX:return &MinMax<1>;
    default:return nullptr;
  }
}

using AggregateArrayFn = std::function<Status(arrow::MemoryPool *pool,
                                              const std::shared_ptr<arrow::Array> &array,
                                              const compute::AggregationOpId &aggregate_op,
                                              const std::vector<int64_t> &boundaries,
                                              std::shared_ptr<arrow::Array> &output_array)>;

template<typename ARROW_T,
    typename = typename std::enable_if<
        arrow::is_number_type<ARROW_T>::value | arrow::is_boolean_type<ARROW_T>::value>::type>
Status AggregateArray(arrow::MemoryPool *pool,
                      const std::shared_ptr<arrow::Array> &array,
                      const compute::AggregationOpId &aggregate_op,
                      const std::vector<int64_t> &boundaries,
                      std::shared_ptr<arrow::Array> &output_array) {
  using BUILDER_T = typename arrow::TypeTraits<ARROW_T>::BuilderType;
  using SCALAR_T = typename arrow::TypeTraits<ARROW_T>::ScalarType;

  BUILDER_T builder(pool);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.Reserve(boundaries.size()));

  arrow::compute::ExecContext exec_ctx(pool);
  arrow::Datum res;

  AggregateFptr aggregate_fptr = PickAggregareFptr(aggregate_op);
  int64_t start = 0;
  for (auto &end: boundaries) {
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(aggregate_fptr(arrow::Datum(array->Slice(start, end - start)), &exec_ctx, &res));
    start = end;
    builder.UnsafeAppend(std::static_pointer_cast<SCALAR_T>(res.scalar())->value);
  }

  RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.Finish(&output_array));

  return Status::OK();
}

AggregateArrayFn PickAggregateArrayFptr(const std::shared_ptr<arrow::DataType> &data_type) {
  switch (data_type->id()) {
    case arrow::Type::BOOL: return &AggregateArray<arrow::BooleanType>;
    case arrow::Type::UINT8: return &AggregateArray<arrow::UInt8Type>;
    case arrow::Type::INT8: return &AggregateArray<arrow::Int8Type>;
    case arrow::Type::UINT16: return &AggregateArray<arrow::UInt16Type>;
    case arrow::Type::INT16: return &AggregateArray<arrow::Int16Type>;
    case arrow::Type::UINT32: return &AggregateArray<arrow::UInt32Type>;
    case arrow::Type::INT32: return &AggregateArray<arrow::Int32Type>;
    case arrow::Type::UINT64: return &AggregateArray<arrow::UInt64Type>;
    case arrow::Type::INT64: return &AggregateArray<arrow::Int64Type>;
    case arrow::Type::FLOAT: return &AggregateArray<arrow::FloatType>;
    case arrow::Type::DOUBLE: return &AggregateArray<arrow::DoubleType>;
    default: return nullptr;
  }
}

template<typename ARROW_T,
    typename = typename std::enable_if<
        arrow::is_number_type<ARROW_T>::value | arrow::is_boolean_type<ARROW_T>::value>::type>
Status make_boundaries(arrow::MemoryPool *pool,
                       const std::shared_ptr<arrow::Array> &array,
                       std::vector<int64_t> &group_boundaries,
                       std::shared_ptr<arrow::Array> &out_array,
                       int64_t *unique_groups) {
  using ARRAY_T = typename arrow::TypeTraits<ARROW_T>::ArrayType;
  using BUILDER_T = typename arrow::TypeTraits<ARROW_T>::BuilderType;
  using C_TYPE = typename ARROW_T::c_type;

  const int64_t len = array->length();

  // reserve a builder for the worst case scenario
  BUILDER_T builder(pool);

  if (len == 0) {
    *unique_groups = 0;
    RETURN_CYLON_STATUS_IF_ARROW_FAILED((builder.Finish(&out_array)));
    return Status::OK();
  }

  group_boundaries.reserve(len);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED((builder.Reserve(len)));

  const std::shared_ptr<ARRAY_T> &index_arr = std::static_pointer_cast<ARRAY_T>(array);
  C_TYPE prev_v = index_arr->Value(0), curr_v;
  builder.UnsafeAppend(prev_v);
  for (int64_t i = 0; i < len; i++) {
    curr_v = index_arr->Value(i);

    if (curr_v > prev_v) {
      group_boundaries.push_back(i);
      prev_v = curr_v;
      builder.UnsafeAppend(curr_v);
    } /*else if (curr_v < prev_v) {
      return cylon::Status(Code::Invalid, "index array not sorted");
    }*/
  }
  group_boundaries.push_back(len);

  group_boundaries.shrink_to_fit();
  RETURN_CYLON_STATUS_IF_ARROW_FAILED((builder.Finish(&out_array)));
  *unique_groups = group_boundaries.size() - 1;

  return Status::OK();
}

using MakeBoundariesFn = std::function<Status(arrow::MemoryPool *pool,
                                              const std::shared_ptr<arrow::Array> &array,
                                              std::vector<int64_t> &group_boundaries,
                                              std::shared_ptr<arrow::Array> &out_array,
                                              int64_t *unique_groups)>;

static inline MakeBoundariesFn pick_make_boundaries_fn(const std::shared_ptr<arrow::DataType> &type) {
  switch (type->id()) {
    case arrow::Type::BOOL: return &make_boundaries<arrow::BooleanType>;
    case arrow::Type::UINT8: return &make_boundaries<arrow::UInt8Type>;
    case arrow::Type::INT8: return &make_boundaries<arrow::Int8Type>;
    case arrow::Type::UINT16: return &make_boundaries<arrow::UInt16Type>;
    case arrow::Type::INT16: return &make_boundaries<arrow::Int16Type>;
    case arrow::Type::UINT32: return &make_boundaries<arrow::UInt32Type>;
    case arrow::Type::INT32: return &make_boundaries<arrow::Int32Type>;
    case arrow::Type::UINT64: return &make_boundaries<arrow::UInt64Type>;
    case arrow::Type::INT64: return &make_boundaries<arrow::Int64Type>;
    case arrow::Type::FLOAT: return &make_boundaries<arrow::FloatType>;
    case arrow::Type::DOUBLE: return &make_boundaries<arrow::DoubleType>;
    default:return nullptr;
  }
}

Status PipelineGroupBy(std::shared_ptr<Table> &table,
                       int32_t index_col,
                       const std::vector<std::pair<int32_t, compute::AggregationOpId>> &aggregations,
                       std::shared_ptr<Table> &output) {
  auto ctx = table->GetContext();
  arrow::MemoryPool *pool = ToArrowPool(ctx);

  const std::shared_ptr<arrow::Table> &a_table = table->get_table();
  const std::shared_ptr<arrow::ChunkedArray> &idx_col = a_table->column(index_col);

  if (idx_col->num_chunks() > 1) {
    return cylon::Status(Code::Invalid, "multiple chunks not supported for pipelined groupby");
  }

  std::vector<int64_t> group_boundaries;
  int64_t unique_groups;
  std::shared_ptr<arrow::Array> out_idx_col;

  MakeBoundariesFn make_boundaries_fn = pick_make_boundaries_fn(idx_col->type());
  RETURN_CYLON_STATUS_IF_FAILED(make_boundaries_fn(pool,
                                                   cylon::util::GetChunkOrEmptyArray(idx_col, 0),
                                                   group_boundaries,
                                                   out_idx_col,
                                                   &unique_groups));

  std::vector<std::shared_ptr<arrow::ChunkedArray>> new_arrays;
  std::vector<std::shared_ptr<arrow::Field>> new_fields;
  int new_cols = (int) (1 + aggregations.size());
  new_arrays.reserve(new_cols);
  new_fields.reserve(new_cols);

  new_arrays.push_back(std::make_shared<arrow::ChunkedArray>(std::move(out_idx_col)));
  new_fields.push_back(a_table->field(index_col));

  for (auto &&agg_op: aggregations) {
    const std::shared_ptr<arrow::ChunkedArray> &agg_col = a_table->column(agg_op.first);
    AggregateArrayFn aggregate_array_fn = PickAggregateArrayFptr(agg_col->type());
    std::shared_ptr<arrow::Array> new_arr;
    RETURN_CYLON_STATUS_IF_FAILED(aggregate_array_fn(pool,
                                                     cylon::util::GetChunkOrEmptyArray(agg_col, 0),
                                                     agg_op.second,
                                                     group_boundaries,
                                                     new_arr));

    new_arrays.push_back(std::make_shared<arrow::ChunkedArray>(std::move(new_arr)));
    new_fields.push_back(a_table->field(agg_op.first)); // todo: prefix name
  }

  const auto &schema = std::make_shared<arrow::Schema>(new_fields);
  std::shared_ptr<arrow::Table> agg_table = arrow::Table::Make(schema, new_arrays);

  output = std::make_shared<Table>(agg_table, ctx);
  return Status::OK();
}

Status PipelineGroupBy(std::shared_ptr<Table> &table,
                              int32_t idx_col,
                              const std::vector<int32_t> &aggregate_cols,
                              const std::vector<compute::AggregationOpId> &aggregate_ops,
                              std::shared_ptr<Table> &output) {
  if (aggregate_cols.size() != aggregate_ops.size()) {
    return Status(Code::Invalid, "aggregate_cols size != aggregate_ops size");
  }

  std::vector<std::pair<int32_t, compute::AggregationOpId>> aggregations;
  aggregations.reserve(aggregate_cols.size());
  for (size_t i = 0; i < aggregate_cols.size(); i++) {
    aggregations.emplace_back(aggregate_cols[i], aggregate_ops[i]);
  }

  return PipelineGroupBy(table, idx_col, aggregations, output);}
}

#endif //CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_PIPELINE_CPP_

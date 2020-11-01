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

#ifndef CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_PIPELINE_HPP_
#define CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_PIPELINE_HPP_

#include <arrow/api.h>
#include <status.hpp>
#include <table.hpp>
#include <ctx/arrow_memory_pool_utils.hpp>
#include <arrow/compute/api.h>

#include "groupby_aggregate_ops.hpp"

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

inline AggregateFptr PickAggregareFptr(const cylon::GroupByAggregationOp aggregation_op) {
  switch (aggregation_op) {
    case SUM: return &Sum;
    case COUNT:return &Count;
    case MIN: return &MinMax<0>;
    case MAX:return &MinMax<1>;
  }
  return nullptr;
}

typedef
arrow::Status (*AggregateArrayFptr)(arrow::MemoryPool *pool,
                                    const std::shared_ptr<arrow::Array> &array,
                                    const cylon::GroupByAggregationOp &aggregate_op,
                                    const std::vector<int64_t> &boundaries,
                                    std::shared_ptr<arrow::Array> &output_array);

template<typename ARROW_T,
    typename = typename std::enable_if<
        arrow::is_number_type<ARROW_T>::value | arrow::is_boolean_type<ARROW_T>::value>::type>
arrow::Status AggregateArray(arrow::MemoryPool *pool,
                             const std::shared_ptr<arrow::Array> &array,
                             const cylon::GroupByAggregationOp &aggregate_op,
                             const std::vector<int64_t> &boundaries,
                             std::shared_ptr<arrow::Array> &output_array) {
  using BUILDER_T = typename arrow::TypeTraits<ARROW_T>::BuilderType;
  using SCALAR_T = typename arrow::TypeTraits<ARROW_T>::ScalarType;
  arrow::Status s;

  BUILDER_T builder(pool);
  if (!(s = builder.Reserve(boundaries.size())).ok()) {
    return s;
  }

  arrow::compute::ExecContext exec_ctx(pool);
  arrow::Datum res;

  AggregateFptr aggregate_fptr = PickAggregareFptr(aggregate_op);
  int64_t start = 0;
  for (auto &end: boundaries) {
    s = aggregate_fptr(arrow::Datum(array->Slice(start, end - start)), &exec_ctx, &res);
    if (!s.ok()) {
      return s;
    }
    start = end;
    builder.UnsafeAppend(std::static_pointer_cast<SCALAR_T>(res.scalar())->value);
  }

  s = builder.Finish(&output_array);
  if (!s.ok()) {
    return s;
  }

  return arrow::Status::OK();
}

AggregateArrayFptr PickAggregateArrayFptr(const std::shared_ptr<cylon::DataType> &val_data_type) {
  switch (val_data_type->getType()) {
    case Type::BOOL: return &AggregateArray<arrow::BooleanType>;
    case Type::UINT8: return &AggregateArray<arrow::UInt8Type>;
    case Type::INT8: return &AggregateArray<arrow::Int8Type>;
    case Type::UINT16: return &AggregateArray<arrow::UInt16Type>;
    case Type::INT16: return &AggregateArray<arrow::Int16Type>;
    case Type::UINT32: return &AggregateArray<arrow::UInt32Type>;
    case Type::INT32: return &AggregateArray<arrow::Int32Type>;
    case Type::UINT64: return &AggregateArray<arrow::UInt64Type>;
    case Type::INT64: return &AggregateArray<arrow::Int64Type>;
    case Type::FLOAT: return &AggregateArray<arrow::FloatType>;
    case Type::DOUBLE: return &AggregateArray<arrow::DoubleType>;
    case Type::HALF_FLOAT:break;
    case Type::STRING:break;
    case Type::BINARY:break;
    case Type::FIXED_SIZE_BINARY:break;
    case Type::DATE32:break;
    case Type::DATE64:break;
    case Type::TIMESTAMP:break;
    case Type::TIME32:break;
    case Type::TIME64:break;
    case Type::INTERVAL:break;
    case Type::DECIMAL:break;
    case Type::LIST:break;
    case Type::EXTENSION:break;
    case Type::FIXED_SIZE_LIST:break;
    case Type::DURATION:break;
  }
  return nullptr;
}

template<typename ARROW_T,
    typename = typename std::enable_if<
        arrow::is_number_type<ARROW_T>::value | arrow::is_boolean_type<ARROW_T>::value>::type>
arrow::Status BuildIndex(arrow::MemoryPool *pool,
                         const std::shared_ptr<arrow::Array> &array,
                         const std::vector<int64_t> &boundaries,
                         std::shared_ptr<arrow::Array> &output_array) {
  using ARRAY_T = typename arrow::TypeTraits<ARROW_T>::ArrayType;
  using BUILDER_T = typename arrow::TypeTraits<ARROW_T>::BuilderType;
  arrow::Status s;

  BUILDER_T builder(pool);
  if (!(s = builder.Reserve(boundaries.size())).ok()) {
    return s;
  }

  const std::shared_ptr<ARRAY_T> &index_arr = std::static_pointer_cast<ARRAY_T>(array);

  for (auto &end: boundaries) {
    builder.UnsafeAppend(index_arr->Value(end - 1));
  }

  s = builder.Finish(&output_array);
  if (!s.ok()) {
    return s;
  }

  return arrow::Status::OK();
}

template<typename IDX_ARROW_T,
    typename = typename std::enable_if<
        arrow::is_number_type<IDX_ARROW_T>::value | arrow::is_boolean_type<IDX_ARROW_T>::value>::type>
Status LocalPipelinedGroupBy(const std::shared_ptr<cylon::Table> &table,
                             const std::vector<cylon::GroupByAggregationOp> &aggregate_ops,
                             std::shared_ptr<cylon::Table> &output) {
  const  std::shared_ptr<arrow::Table> &a_table = table->get_table();
  const  std::shared_ptr<arrow::ChunkedArray> &idx_col = a_table->column(0);

  if (idx_col->num_chunks() > 1) {
    return cylon::Status(Code::Invalid, "multiple chunks not supported for pipelined groupby");
  }

  auto ctx = table->GetContext();

  arrow::MemoryPool *memory_pool = cylon::ToArrowPool(ctx);

  using IDX_C_T = typename arrow::TypeTraits<IDX_ARROW_T>::CType;
  using IDX_ARRAY_T = typename arrow::TypeTraits<IDX_ARROW_T>::ArrayType;

  arrow::Status s;

  std::vector<int64_t> boundaries;
  const int64_t len = idx_col->length();
  boundaries.reserve((int64_t) len * 0.5);

  const std::shared_ptr<IDX_ARRAY_T> &index_arr = std::static_pointer_cast<IDX_ARRAY_T>
      (idx_col->chunk(0));

  IDX_C_T prev_v = index_arr->Value(0), curr_v;
  for (int64_t i = 0; i < len; i++) {
    curr_v = index_arr->Value(i);

    if (curr_v > prev_v) {
      boundaries.push_back(i);
      prev_v = curr_v;
    } else if (curr_v < prev_v) {
      return cylon::Status(Code::Invalid, "index array not sorted");
    }
  }
  boundaries.push_back(len);

  std::vector<std::shared_ptr<arrow::Array>> out_arrays;

  // build index
   std::shared_ptr<arrow::Array> out_idx;
  s = BuildIndex<IDX_ARROW_T>(memory_pool, idx_col->chunk(0), boundaries, out_idx);
  if (!s.ok()) {
    return cylon::Status(static_cast<int>(s.code()), s.message());
  }
  out_arrays.push_back(out_idx);

  for (size_t i = 0; i < aggregate_ops.size(); i++) {
    const  std::shared_ptr<DataType> &val_type = table->GetColumn(i + 1)->GetDataType();
    AggregateArrayFptr aggregate_array_fn = PickAggregateArrayFptr(val_type);
    std::shared_ptr<arrow::Array> out_val;
    s = aggregate_array_fn(memory_pool, a_table->column(i + 1)->chunk(0), aggregate_ops[i], boundaries, out_val);

    if (!s.ok()) {
      return cylon::Status(static_cast<int>(s.code()), s.message());
    }

    out_arrays.push_back(out_val);
  }

   std::shared_ptr<arrow::Table> a_output = arrow::Table::Make(a_table->schema(), out_arrays);

  return cylon::Table::FromArrowTable(ctx, a_output, output);
}

}

#endif //CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_PIPELINE_HPP_

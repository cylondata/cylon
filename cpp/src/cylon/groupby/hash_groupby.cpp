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

#include <arrow/api.h>
#include <arrow/visitor_inline.h>
#include <arrow/compute/api.h>
#include <chrono>
#include <glog/logging.h>

#include "cylon/arrow/arrow_comparator.hpp"
#include "cylon/ctx/arrow_memory_pool_utils.hpp"
#include "cylon/util/macros.hpp"
#include "cylon/groupby/hash_groupby.hpp"
#include "cylon/thridparty/flat_hash_map/bytell_hash_map.hpp"

namespace cylon {

/**
 * Local hash group-by implementation
 *
 * Implemented in 2 stages.
 *
 *  1. Create groups - assigns a unique ID (int64) to groups derived from index columns, and keep group IDs in a vector
 *
 *  2. Aggregate values based on the group ID
 */

// -----------------------------------------------------------------------------

/**
 * create unique group IDs based on index columns of a table
 *
 * algorithm - Use a hash map of <row index(int64), group ID(int64)> to assign unique IDs. Make a composite hash using
 * index columns and use this composite hashes as a custom hash function to the hash map (cylon::TableRowIndexHash).
 * If there are hash conflicts, iterate through the row and determine the equality (cylon::TableRowIndexComparator).
 *
 * Hash map insertion result outputs a <iterator,bool> pair. If the key is unique, the iterator would return the
 * position at which it was inserted and a bool true. If the key is already in the map, the iterator would indicate the
 * position of the existing key and a bool false.
 * So, if by looking at the bool, we can determine if the current index's values constitute a unique group. If we keep a
 * unique_groups counter and insert its value to the map as the group ID and increment it when we receive a true from the
 * insertion result.
 *
 * Following is an example of this logic
 *
 * row_idx, A, B, C
 * 0, 20, 20, x
 * 1, 10, 10, y
 * 2, 20, 10, z
 * 3, 10, 10, x
 * 4, 20, 20, y
 * 5, 30, 20, z
 *
 * grouping by [A, B]
 *
 * then,
 * row_idx, group_id
 * 0, 0
 * 1, 1
 * 2, 2
 * 3, 1
 * 4, 0
 * 5, 3
 * unique_groups = 4
 *
 *
 * Additionally, we create an arrow::Array with the indices of the unique groups
 *
 * @param pool
 * @param atable
 * @param idx_cols
 * @param group_ids
 * @param group_filter
 * @param unique_groups
 * @return
 */
// todo handle chunked arrays
static Status make_groups(arrow::MemoryPool *pool,
                          const std::shared_ptr<arrow::Table> &atable,
                          const std::vector<int> &idx_cols,
                          std::vector<int64_t> *group_ids,
                          std::shared_ptr<arrow::Array> *group_filter,
                          int64_t *unique_groups) {
  const int64_t num_rows = atable->num_rows();

  // if empty, return 
  if (num_rows == 0) {
    CYLON_ASSIGN_OR_RAISE(*group_filter, arrow::MakeArrayOfNull(arrow::int64(), 0, pool))
    return Status::OK();
  }

  std::unique_ptr<TableRowIndexEqualTo> comp;
  RETURN_CYLON_STATUS_IF_FAILED(TableRowIndexEqualTo::Make(atable, idx_cols, &comp));

  std::unique_ptr<TableRowIndexHash> hash;
  RETURN_CYLON_STATUS_IF_FAILED(TableRowIndexHash::Make(atable, idx_cols, &hash));

  ska::bytell_hash_map<int64_t, int64_t, TableRowIndexHash, TableRowIndexEqualTo>
      hash_map(num_rows, *hash, *comp);

  group_ids->reserve(num_rows);
  arrow::Int64Builder filter_build(pool);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED((filter_build.Reserve(num_rows)));

  int64_t unique = 0;
  for (int64_t i = 0; i < num_rows; i++) {
    const auto &res = hash_map.emplace(i, unique);
    if (res.second) { // this was a unique group
      group_ids->emplace_back(unique);
      unique++;
      filter_build.UnsafeAppend(i);
    } else {
      group_ids->emplace_back(res.first->second);
    }
  }

  RETURN_CYLON_STATUS_IF_ARROW_FAILED((filter_build.Finish(group_filter)));
  *unique_groups = unique;
  return Status::OK();
}

/**
 * aggregate operation execution based on the unique group IDs.
 * @tparam aggOp AggregationOpId
 * @tparam ARROW_T
 * @param pool
 * @param arr
 * @param field
 * @param group_ids
 * @param unique_groups
 * @param agg_array aggregated arrow::Array
 * @param agg_field aggregated arrow::Array schema field
 * @return Status
 */
// todo handle chunked arrays
template<compute::AggregationOpId aggOp, typename ARROW_T,
    typename = arrow::enable_if_has_c_type<ARROW_T>>
inline Status aggregate(arrow::MemoryPool *pool,
                        const std::shared_ptr<arrow::Table> &table,
                        int col_idx,
                        const compute::AggregationOp &agg_op,
                        const std::vector<int64_t> &group_ids,
                        int64_t unique_groups,
                        std::shared_ptr<arrow::Array> *agg_array,
                        std::shared_ptr<arrow::Field> *agg_field) {
  using C_TYPE = typename ARROW_T::c_type;

  using State = typename compute::KernelTraits<aggOp, C_TYPE>::State;
  using ResultT = typename compute::KernelTraits<aggOp, C_TYPE>::ResultT;
  using Options = typename compute::KernelTraits<aggOp, C_TYPE>::Options;

  auto update_field = [&](const std::shared_ptr<arrow::DataType> &ret_type) {
    const char *prefix = compute::KernelTraits<aggOp, C_TYPE>::name();
    // in dist group by, aggregate may be called multiple times on the same col. Check issue #387
    const auto &field = table->schema()->field(col_idx);
    if (field->name().find(prefix) == std::string::npos) {
      *agg_field = std::make_shared<arrow::Field>(std::string(prefix) + field->name(), ret_type);
    } else {
      *agg_field = std::make_shared<arrow::Field>(field->name(), ret_type);
    }
  };

  const int64_t len = table->num_rows();

  if (len == 0) {
    auto ret_type =
        arrow::TypeTraits<typename arrow::CTypeTraits<ResultT>::ArrowType>::type_singleton();
    CYLON_ASSIGN_OR_RAISE(*agg_array, arrow::MakeArrayOfNull(ret_type, 0, pool))
    update_field(ret_type);

    return Status::OK();
  }

  if (len != (int64_t) group_ids.size()) {
    return {Code::Invalid, "group IDs != array length"};
  }

  const auto &kernel = compute::CreateAggregateKernel<aggOp, C_TYPE>();
  if (kernel == nullptr) {
    return {Code::Invalid, "unsupported aggregate op"};
  }

  compute::KernelOptions *options = agg_op.options.get();
  if (options != nullptr) {
    kernel->Setup(options);
  } else {
    Options opt;
    kernel->Setup(&opt);
  }

  State initial_state;
  kernel->InitializeState(&initial_state); // initialize state

  // initialize aggregate states by copying initial state
  std::vector<State> agg_states(unique_groups, initial_state);

  const auto &arr = table->column(col_idx)->chunk(0);
  int64_t i = 0;
  arrow::VisitArrayDataInline<ARROW_T>(*arr->data(),
                                       [&](const C_TYPE &val) {
                                         kernel->Update(&val, &agg_states[group_ids[i]]);
                                         i++;
                                       },
                                       [&]() {
                                         i++;
                                       });

  // need to create a builder from the ResultT, which is a C type
  using RESULT_ARROW_T = typename arrow::CTypeTraits<ResultT>::ArrowType;
  using BUILDER_T = typename arrow::TypeTraits<RESULT_ARROW_T>::BuilderType;
  BUILDER_T builder(pool);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.Reserve(unique_groups));
  for (int64_t group_id = 0; group_id < unique_groups; group_id++) {
    ResultT res;
    kernel->Finalize(&agg_states[group_id], &res);
    builder.UnsafeAppend(res);
  }

  RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.Finish(agg_array));

  update_field(arrow::TypeTraits<RESULT_ARROW_T>::type_singleton());
  return Status::OK();
}

template<typename ARROW_T, typename = arrow::enable_if_has_c_type<ARROW_T>>
inline Status resolve_op(arrow::MemoryPool *pool,
                         const std::shared_ptr<arrow::Table> &table,
                         int col_idx,
                         const compute::AggregationOp &agg_op,
                         const std::vector<int64_t> &group_ids,
                         int64_t unique_groups,
                         std::shared_ptr<arrow::Array> *agg_array,
                         std::shared_ptr<arrow::Field> *agg_field) {
  switch (agg_op.id) {
    case compute::SUM:
      return aggregate<compute::SUM, ARROW_T>(pool, table, col_idx, agg_op, group_ids,
                                              unique_groups, agg_array, agg_field);
    case compute::COUNT:
      return aggregate<compute::COUNT, ARROW_T>(pool, table, col_idx, agg_op, group_ids,
                                                unique_groups, agg_array, agg_field);
    case compute::MIN:
      return aggregate<compute::MIN, ARROW_T>(pool, table, col_idx, agg_op, group_ids,
                                              unique_groups, agg_array, agg_field);
    case compute::MAX:
      return aggregate<compute::MAX, ARROW_T>(pool, table, col_idx, agg_op, group_ids,
                                              unique_groups, agg_array, agg_field);
    case compute::MEAN:
      return aggregate<compute::MEAN, ARROW_T>(pool, table, col_idx, agg_op, group_ids,
                                               unique_groups, agg_array, agg_field);
    case compute::VAR:
      return aggregate<compute::VAR, ARROW_T>(pool, table, col_idx, agg_op, group_ids,
                                              unique_groups, agg_array, agg_field);
    case compute::NUNIQUE:
      return aggregate<compute::NUNIQUE, ARROW_T>(pool, table, col_idx, agg_op, group_ids,
                                                  unique_groups, agg_array, agg_field);
    case compute::QUANTILE:
      return aggregate<compute::QUANTILE, ARROW_T>(pool, table, col_idx, agg_op, group_ids,
                                                   unique_groups, agg_array, agg_field);
    case compute::STDDEV:
      return aggregate<compute::STDDEV, ARROW_T>(pool, table, col_idx, agg_op, group_ids,
                                                 unique_groups, agg_array, agg_field);
    default: return {Code::Invalid, "Unsupported aggregation operation"};
  }
}

inline Status do_aggregate(arrow::MemoryPool *pool,
                           const std::shared_ptr<arrow::Table> &table,
                           int col_idx,
                           const compute::AggregationOp &agg_op,
                           const std::vector<int64_t> &group_ids,
                           int64_t unique_groups,
                           std::shared_ptr<arrow::Array> *agg_array,
                           std::shared_ptr<arrow::Field> *agg_field) {
  switch (table->schema()->field(col_idx)->type()->id()) {
    case arrow::Type::BOOL:
      return resolve_op<arrow::BooleanType>(pool, table, col_idx, agg_op, group_ids,
                                            unique_groups, agg_array, agg_field);
    case arrow::Type::UINT8:
      return resolve_op<arrow::UInt8Type>(pool, table, col_idx, agg_op, group_ids,
                                          unique_groups, agg_array, agg_field);
    case arrow::Type::INT8:
      return resolve_op<arrow::Int8Type>(pool, table, col_idx, agg_op, group_ids,
                                         unique_groups, agg_array, agg_field);
    case arrow::Type::UINT16:
      return resolve_op<arrow::UInt16Type>(pool, table, col_idx, agg_op, group_ids,
                                           unique_groups, agg_array, agg_field);
    case arrow::Type::INT16:
      return resolve_op<arrow::Int16Type>(pool, table, col_idx, agg_op, group_ids,
                                          unique_groups, agg_array, agg_field);
    case arrow::Type::UINT32:
      return resolve_op<arrow::UInt32Type>(pool, table, col_idx, agg_op, group_ids,
                                           unique_groups, agg_array, agg_field);
    case arrow::Type::INT32:
      return resolve_op<arrow::Int32Type>(pool, table, col_idx, agg_op, group_ids,
                                          unique_groups, agg_array, agg_field);
    case arrow::Type::UINT64:
      return resolve_op<arrow::UInt64Type>(pool, table, col_idx, agg_op, group_ids,
                                           unique_groups, agg_array, agg_field);
    case arrow::Type::INT64:
      return resolve_op<arrow::Int64Type>(pool, table, col_idx, agg_op, group_ids,
                                          unique_groups, agg_array, agg_field);
    case arrow::Type::FLOAT:
      return resolve_op<arrow::FloatType>(pool, table, col_idx, agg_op, group_ids,
                                          unique_groups, agg_array, agg_field);
    case arrow::Type::DOUBLE:
      return resolve_op<arrow::DoubleType>(pool, table, col_idx, agg_op, group_ids,
                                           unique_groups, agg_array, agg_field);
    case arrow::Type::DATE32:
      return resolve_op<arrow::Date32Type>(pool, table, col_idx, agg_op, group_ids,
                                           unique_groups, agg_array, agg_field);
    case arrow::Type::DATE64:
      return resolve_op<arrow::Date64Type>(pool, table, col_idx, agg_op, group_ids,
                                           unique_groups, agg_array, agg_field);
    case arrow::Type::TIMESTAMP:
      return resolve_op<arrow::TimestampType>(pool, table, col_idx, agg_op, group_ids,
                                              unique_groups, agg_array, agg_field);
    case arrow::Type::TIME32:
      return resolve_op<arrow::Time32Type>(pool, table, col_idx, agg_op, group_ids,
                                           unique_groups, agg_array, agg_field);
    case arrow::Type::TIME64:
      return resolve_op<arrow::Time64Type>(pool, table, col_idx, agg_op, group_ids,
                                           unique_groups, agg_array, agg_field);
    default:
      return {Code::NotImplemented, "Unsupported type for aggregations "
          + table->schema()->field(col_idx)->type()->ToString()};
  }
}

Status HashGroupBy(const std::shared_ptr<Table> &table,
                   const std::vector<int32_t> &idx_cols,
                   const std::vector<std::pair<int32_t, std::shared_ptr<compute::AggregationOp>>> &aggregations,
                   std::shared_ptr<Table> &output) {
#ifdef CYLON_DEBUG
  auto t1 = std::chrono::steady_clock::now();
#endif
  const auto &ctx = table->GetContext();
  arrow::MemoryPool *pool = ToArrowPool(ctx);

  std::shared_ptr<arrow::Table> atable = table->get_table();
  COMBINE_CHUNKS_RETURN_CYLON_STATUS(atable, pool);
#ifdef CYLON_DEBUG
  auto t2 = std::chrono::steady_clock::now();
#endif
  std::vector<int64_t> group_ids;
  int64_t unique_groups = 0;
  std::shared_ptr<arrow::Array> group_filter;
  RETURN_CYLON_STATUS_IF_FAILED(make_groups(pool, atable, idx_cols, &group_ids, &group_filter,
                                            &unique_groups));
#ifdef CYLON_DEBUG
  auto t3 = std::chrono::steady_clock::now();
#endif
  std::vector<std::shared_ptr<arrow::ChunkedArray>> new_arrays;
  std::vector<std::shared_ptr<arrow::Field>> new_fields;
  int new_cols = (int) (idx_cols.size() + aggregations.size());
  new_arrays.reserve(new_cols);
  new_fields.reserve(new_cols);

  //first filter idx cols
  for (auto &&i: idx_cols) {
    CYLON_ASSIGN_OR_RAISE(auto res, arrow::compute::Take(atable->column(i), group_filter));
    new_arrays.emplace_back(res.chunked_array());
    new_fields.emplace_back(atable->field(i));
  }

  // then aggregate other cols
  for (auto &&p: aggregations) {
    std::shared_ptr<arrow::Array> new_arr;
    std::shared_ptr<arrow::Field> new_field;
    RETURN_CYLON_STATUS_IF_FAILED(do_aggregate(pool, atable, p.first, *p.second, group_ids,
                                               unique_groups, &new_arr, &new_field));
    new_arrays.push_back(std::make_shared<arrow::ChunkedArray>(std::move(new_arr)));
    new_fields.push_back(std::move(new_field));
  }

  auto schema = arrow::schema(std::move(new_fields));
  auto agg_table = arrow::Table::Make(std::move(schema), std::move(new_arrays));

  output = std::make_shared<Table>(ctx, std::move(agg_table));
#ifdef CYLON_DEBUG
  auto t4 = std::chrono::steady_clock::now();
  LOG(INFO) << "hash groupby setup:" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
            << " make_groups:" << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
            << " aggregate:" << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()
            << " total:" << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t1).count();
#endif
  return Status::OK();
}

Status HashGroupBy(const std::shared_ptr<Table> &table,
                   const std::vector<int32_t> &idx_cols,
                   const std::vector<std::pair<int32_t, compute::AggregationOpId>> &aggregate_cols,
                   std::shared_ptr<Table> &output) {
  std::vector<std::pair<int32_t, std::shared_ptr<compute::AggregationOp>>> aggregations;
  aggregations.reserve(aggregate_cols.size());
  for (auto &&p:aggregate_cols) {
    // create AggregationOp with nullptr options
    aggregations.emplace_back(p.first, std::make_shared<compute::AggregationOp>(p.second));
  }

  return HashGroupBy(table, idx_cols, aggregations, output);
}

Status HashGroupBy(std::shared_ptr<Table> &table,
                   const std::vector<int32_t> &idx_cols,
                   const std::vector<int32_t> &aggregate_cols,
                   const std::vector<compute::AggregationOpId> &aggregate_ops,
                   std::shared_ptr<Table> &output) {
  if (aggregate_cols.size() != aggregate_ops.size()) {
    return {Code::Invalid, "aggregate_cols size != aggregate_ops size"};
  }

  std::vector<std::pair<int32_t, std::shared_ptr<compute::AggregationOp>>> aggregations;
  aggregations.reserve(aggregate_cols.size());
  for (size_t i = 0; i < aggregate_cols.size(); i++) {
    aggregations.emplace_back(aggregate_cols[i], std::make_shared<compute::AggregationOp>(aggregate_ops[i]));
  }

  return HashGroupBy(table, idx_cols, aggregations, output);
}

Status HashGroupBy(std::shared_ptr<Table> &table,
                   int32_t idx_col,
                   const std::vector<int32_t> &aggregate_cols,
                   const std::vector<compute::AggregationOpId> &aggregate_ops,
                   std::shared_ptr<Table> &output) {
  return HashGroupBy(table, std::vector<int32_t>{idx_col}, aggregate_cols, aggregate_ops, output);
}

}
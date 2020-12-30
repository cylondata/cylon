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

#include "hash_groupby.hpp"
#include <arrow/arrow_comparator.hpp>
#include "thridparty/flat_hash_map/bytell_hash_map.hpp"

namespace cylon {

static Status make_groups(arrow::MemoryPool *pool,
                          const std::shared_ptr<arrow::Table> &atable,
                          const std::vector<int> &idx_cols,
                          std::vector<int64_t> &group_ids,
                          std::shared_ptr<arrow::Array> &group_filter,
                          int64_t *unique_groups) {
  TableRowIndexHash hash(atable, idx_cols);
  TableRowIndexComparator comp(atable, idx_cols);

  const int64_t num_rows = atable->num_rows();

  ska::bytell_hash_map<int64_t, int64_t, TableRowIndexHash, TableRowIndexComparator>
      hash_map(num_rows, hash, comp);

  group_ids.reserve(num_rows);
  arrow::BooleanBuilder filter_build(pool);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED((filter_build.Reserve(num_rows)))

  int64_t unique = 0;
  for (int64_t i = 0; i < num_rows; i++) {
    const auto &res = hash_map.insert(std::make_pair(i, unique));
    if (res.second) { // this was a unique group
      group_ids.push_back(unique);
      unique++;
    } else {
      group_ids.push_back(res.first->second);
    }
    filter_build.UnsafeAppend(res.second);
  }

  RETURN_CYLON_STATUS_IF_ARROW_FAILED((filter_build.Finish(&group_filter)))
  *unique_groups = unique;
  return Status::OK();
}


static AggregationOp pick_aggregation_op(const std::shared_ptr<arrow::DataType> &val_data_type,
                                         const cylon::compute::AggregationOp op) {
  switch (val_data_type->id()) {
    case arrow::Type::BOOL: return resolve_op<arrow::BooleanType>(op);
    case arrow::Type::UINT8: return resolve_op<arrow::UInt8Type>(op);
    case arrow::Type::INT8: return resolve_op<arrow::Int8Type>(op);
    case arrow::Type::UINT16: return resolve_op<arrow::UInt16Type>(op);
    case arrow::Type::INT16: return resolve_op<arrow::Int16Type>(op);
    case arrow::Type::UINT32: return resolve_op<arrow::UInt32Type>(op);
    case arrow::Type::INT32: return resolve_op<arrow::Int32Type>(op);
    case arrow::Type::UINT64: return resolve_op<arrow::UInt64Type>(op);
    case arrow::Type::INT64: return resolve_op<arrow::Int64Type>(op);
    case arrow::Type::FLOAT: return resolve_op<arrow::FloatType>(op);
    case arrow::Type::DOUBLE: return resolve_op<arrow::DoubleType>(op);
    default: return nullptr;
  }
}

Status HashGroupBy(const std::shared_ptr<Table> &table,
                   const std::vector<int32_t> &idx_cols,
                   const std::vector<std::pair<int64_t, compute::AggregationOp>> &aggregate_cols,
                   std::shared_ptr<Table> &output) {
  std::vector<int64_t> group_ids;
  int64_t unique_groups;
  std::shared_ptr<arrow::Array> group_filter;
  auto ctx = table->GetContext();
  arrow::MemoryPool *pool = ToArrowPool(ctx);

  std::shared_ptr<arrow::Table> atable = table->get_table();
  if (atable->column(0)->num_chunks() > 1) { // todo: make this work with chunked arrays
    const auto &res = atable->CombineChunks(pool);
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(res.status())
    atable = res.ValueOrDie();
  }

  RETURN_CYLON_STATUS_IF_FAILED(make_groups(pool, atable, idx_cols, group_ids, group_filter, &unique_groups))

  std::vector<std::shared_ptr<arrow::ChunkedArray>> new_arrays;
  std::vector<std::shared_ptr<arrow::Field>> new_fields;
  int new_cols = (int) (idx_cols.size() + aggregate_cols.size());
  new_arrays.reserve(new_cols);
  new_fields.reserve(new_cols);

  //first filter idx cols
  for (auto &&i: idx_cols) {
    const arrow::Result<arrow::Datum> &res = arrow::compute::Filter(atable->column(i), group_filter);
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(res.status())
    new_arrays.push_back(res.ValueOrDie().chunked_array());
    new_fields.push_back(atable->field(i));
  }

  // then aggregate other cols
  for (auto &&p: aggregate_cols) {
    std::shared_ptr<arrow::Array> new_arr;
    std::shared_ptr<arrow::Field> new_field;
    const AggregationOp &agg_op = pick_aggregation_op(atable->field(p.first)->type(), p.second);

    RETURN_CYLON_STATUS_IF_FAILED(agg_op(pool,
                                         atable->column(p.first)->chunk(0),
                                         atable->field(p.first),
                                         group_ids,
                                         unique_groups,
                                         new_arr,
                                         new_field))
    new_arrays.push_back(std::make_shared<arrow::ChunkedArray>(std::move(new_arr)));
    new_fields.push_back(std::move(new_field));
  }

  const auto &schema = std::make_shared<arrow::Schema>(new_fields);
  std::shared_ptr<arrow::Table> agg_table = arrow::Table::Make(schema, new_arrays);

  output = std::make_shared<Table>(agg_table, ctx);
  return Status::OK();
}

}
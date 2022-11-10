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


#include "hash_window.hpp"

#include "cylon/thridparty/flat_hash_map/bytell_hash_map.hpp"
#include <arrow/api.h>
#include <arrow/visitor_inline.h>
#include <arrow/compute/api.h>
#include <chrono>
#include <glog/logging.h>

#include "cylon/arrow/arrow_comparator.hpp"
#include "cylon/ctx/arrow_memory_pool_utils.hpp"
#include "cylon/util/macros.hpp"
#include <cylon/groupby/hash_groupby.hpp>

namespace cylon {
namespace windowing {

Status HashWindow(const config::WindowConfig &window_config,
                  const std::shared_ptr<Table> &table,
                  const std::vector<int32_t> &idx_cols,
                  const std::vector<std::pair<int32_t, compute::AggregationOpId>> &aggregate_cols,
                  std::shared_ptr<Table> &output) {
  std::vector<std::pair<int32_t, std::shared_ptr<compute::AggregationOp>>> aggregations;
  aggregations.reserve(aggregate_cols.size());
  for (auto &&p:aggregate_cols) {
    // create AggregationOp with nullptr options
    aggregations.emplace_back(p.first, compute::MakeAggregationOpFromID(p.second));
  }

  return HashWindow(window_config, table, idx_cols, aggregations, output);
}


/**
 * Hash group-by operation by using <col_index, AggregationOp> pairs
 * NOTE: Nulls in the value columns will be ignored!
 * @param table
 * @param idx_cols
 * @param aggregations
 * @param output
 * @return
 */
Status HashWindow(const config::WindowConfig &window_config,
                  const std::shared_ptr<Table> &table,
                  const std::vector<int32_t> &idx_cols,
                  const std::vector<std::pair<int32_t, std::shared_ptr<compute::AggregationOp>>> &aggregations,
                  std::shared_ptr<Table> &output) {

#ifdef CYLON_DEBUG
  auto t1 = std::chrono::steady_clock::now();
#endif
  const auto &ctx = table->GetContext();
  arrow::MemoryPool *pool = ToArrowPool(ctx);

  std::shared_ptr<arrow::Table> atable = table->get_table();

  std::vector<std::shared_ptr<Table>> slices;

  //slide arrays
  if (window_config.GetObservations() > 0 && atable->num_rows() > 0) {
    SlicesByObservations(window_config, table, slices, pool);
  }

  std::vector<std::shared_ptr<Table>> agg_slices;

  if (!slices.empty()) {

    for (const auto &slice : slices) {
      //call hash group by for each slice and then combine slices
      std::shared_ptr<cylon::Table> output;
      RETURN_CYLON_STATUS_IF_FAILED(cylon::HashGroupBy(slice, idx_cols, aggregations, output));
      agg_slices.emplace_back(std::move(output));
    }
  }

  if (!agg_slices.empty()) {
    RETURN_CYLON_STATUS_IF_FAILED(Merge(agg_slices, output));
  }

  return cylon::Status::OK();
}

Status CreateEmptyTableAndMerge(const std::shared_ptr<Table> *sliced_table,
                                const std::shared_ptr<arrow::Schema> &schema,
                                std::shared_ptr<Table> &output,
                                arrow::MemoryPool *pool, int64_t num_rows) {
  std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;
  arrays.reserve(schema->num_fields());

  auto ctx = sliced_table->get()->GetContext();

  for (int i = 0; i < schema->num_fields(); i++) {
    const auto &t = schema->field(i)->type();
    CYLON_ASSIGN_OR_RAISE(auto arr, arrow::MakeArrayOfNull(t, 0, pool))
    arrays.emplace_back(std::make_shared<arrow::ChunkedArray>(std::move(arr)));
  }

  if (num_rows > 0) {

    std::shared_ptr<arrow::Table> arrow_empty_table = arrow::Table::Make(schema, std::move(arrays), num_rows);
    std::shared_ptr<Table> cylonEmptyTable1;
    auto status1 = Table::FromArrowTable(ctx, std::move(arrow_empty_table), cylonEmptyTable1);

    if (!status1.is_ok()) {
      return status1;
    }

    std::shared_ptr<Table> concat;

    std::shared_ptr<Table> cylonSlicedTable;

    RETURN_CYLON_STATUS_IF_FAILED(Table::FromArrowTable(ctx, sliced_table->get()->get_table(), cylonSlicedTable));
    RETURN_CYLON_STATUS_IF_FAILED(Merge({cylonEmptyTable1, cylonSlicedTable}, concat));

    output = std::move(cylonSlicedTable);
  } else {
    output = *sliced_table;
  }

  return cylon::Status::OK();
}

Status SlicesByObservations(const config::WindowConfig &window_config,
                            const std::shared_ptr<Table> &table,
                            std::vector<std::shared_ptr<Table>> &output,
                            arrow::MemoryPool *pool) {
  auto observations = window_config.GetObservations();

  std::shared_ptr<arrow::Table> atable = table->get_table();

  if (observations > atable->num_rows()) { //bound observations to number of rows
    observations = atable->num_rows();
  }

  for (int64_t i = 0; i < atable->num_rows(); i++) {

    auto adjusted_observations = observations - 1;

    if ((i + 1) < observations ) { //prefill table with nulls
      auto num_rows_to_add = adjusted_observations - i;
      auto slice_start =  i - num_rows_to_add;

      std::shared_ptr<Table> merge_output;

      std::shared_ptr<Table> cylon_output;

      if (slice_start <= 0) {  //check if neg

        auto sliceSize = observations - num_rows_to_add;
        auto slice = atable->Slice(0, sliceSize);
        RETURN_CYLON_STATUS_IF_FAILED(Table::FromArrowTable(table->GetContext(), slice, cylon_output));
        CreateEmptyTableAndMerge(&cylon_output, atable->schema(), merge_output, pool,
                                   num_rows_to_add);
        output.emplace_back(std::move(merge_output));

      } else {

        auto slice = atable->Slice(i - num_rows_to_add, observations);
        RETURN_CYLON_STATUS_IF_FAILED(Table::FromArrowTable(table->GetContext(),
                                                            slice,
                                                            cylon_output));
        CreateEmptyTableAndMerge(&cylon_output, atable->schema(), merge_output, pool,
                                 num_rows_to_add);
        output.emplace_back(std::move(merge_output));
      }

    } else {
      std::shared_ptr<Table> cylonTable;
      auto status = Table::FromArrowTable(table->GetContext(), atable->Slice(i - adjusted_observations, observations),
                            cylonTable);
      if (status.is_ok()) {
        output.emplace_back(std::move(cylonTable));
      }
    }
  }

  return cylon::Status::OK();

}


Status SlicesByOffset(const config::WindowConfig &window_config,
                      const std::shared_ptr<arrow::Table> &table,
                      std::vector<Table> &output) {
  return cylon::Status::OK();
}

Status HashWindow(const config::WindowConfig &window_config,
                  std::shared_ptr<Table> &table,
                  int32_t idx_col,
                  const std::vector<int32_t> &aggregate_cols,
                  const std::vector<compute::AggregationOpId> &aggregate_ops,
                  std::shared_ptr<Table> &output) {

  return cylon::Status::OK();
}

}
}

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

#ifndef CYLON_SRC_CYLON_WINDOW_HASH_WINDOW_HPP_

#include <cylon/table.hpp>
#include <cylon/compute/aggregate_kernels.hpp>
#include "window_config.hpp"


namespace cylon {
namespace windowing {

/**
 * Hash group-by operation by using <col_index, AggregationOpId> pairs
 * NOTE: Nulls in the value columns will be ignored!
 * @param table
 * @param idx_cols
 * @param aggregate_cols
 * @param output
 * @return
 */
Status HashWindow(const config::WindowConfig &window_config,
                  const std::shared_ptr<Table> &table,
                  const std::vector<int32_t> &idx_cols,
                  const std::vector<std::pair<int32_t, compute::AggregationOpId>> &aggregate_cols,
                  std::shared_ptr<Table> &output);

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
                  std::shared_ptr<Table> &output);

/**
 * Returns a vector of windows sliced by observation size
 * @param window_config
 * @param table
 * @param output
 * @param pool
 * @return
 */
Status SlicesByObservations(const config::WindowConfig &window_config,
                        const std::shared_ptr<Table> &table,
                        std::vector<std::shared_ptr<Table>> &output,
                            arrow::MemoryPool *pool);

Status CreateEmptyTableAndMerge(const std::shared_ptr<Table> *sliced_table,
                               const std::shared_ptr<arrow::Schema> &schema,
                               std::shared_ptr<Table> &output,
                               arrow::MemoryPool *pool, int64_t num_rows);

Status SlicesByOffset(const config::WindowConfig &window_config,
                            const std::shared_ptr<arrow::Table> &table,
                            std::vector<Table> &output);
/**
 * Hash group-by operation by using AggregationOpId vector
 * NOTE: Nulls in the value columns will be ignored!
 * @param table
 * @param idx_cols
 * @param aggregate_cols
 * @param aggregate_ops
 * @param output
 * @return
 */
Status HashWindow(const config::WindowConfig &window_config,
                  std::shared_ptr<Table> &table,
                  const std::vector<int32_t> &idx_cols,
                  const std::vector<int32_t> &aggregate_cols,
                  const std::vector<compute::AggregationOpId> &aggregate_ops,
                  std::shared_ptr<Table> &output);

Status HashWindow(const config::WindowConfig &window_config,
                  std::shared_ptr<Table> &table,
                  int32_t idx_col,
                  const std::vector<int32_t> &aggregate_cols,
                  const std::vector<compute::AggregationOpId> &aggregate_ops,
                  std::shared_ptr<Table> &output);

}
}

#endif //CYLON_SRC_CYLON_WINDOW_HASH_WINDOW_HPP_

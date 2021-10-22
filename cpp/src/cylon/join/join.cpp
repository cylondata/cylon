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

#include <glog/logging.h>
#include <chrono>
#include <string>

#include <cylon/join/join.hpp>
#include <cylon/join/join_utils.hpp>
#include <cylon/join/hash_join.hpp>
#include <cylon/join/sort_join.hpp>
#include <cylon/arrow/arrow_kernels.hpp>
#include <cylon/arrow/arrow_comparator.hpp>

namespace cylon {
namespace join {

Status JoinTables(const std::vector<std::shared_ptr<arrow::Table>> &left_tabs,
                  const std::vector<std::shared_ptr<arrow::Table>> &right_tabs,
                  const config::JoinConfig &join_config,
                  std::shared_ptr<arrow::Table> *joined_table,
                  arrow::MemoryPool *memory_pool) {
  CYLON_ASSIGN_OR_RAISE(auto left_tab, arrow::ConcatenateTables(left_tabs,
                                                                arrow::ConcatenateTablesOptions::Defaults(),
                                                                memory_pool))
  CYLON_ASSIGN_OR_RAISE(auto right_tab, arrow::ConcatenateTables(right_tabs,
                                                                 arrow::ConcatenateTablesOptions::Defaults(),
                                                                 memory_pool));

  CYLON_ASSIGN_OR_RAISE(auto left_tab_combined, left_tab->CombineChunks(memory_pool));

  if (left_tabs.size() > 1) {
    for (const auto &t: left_tabs) {
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(cylon::util::free_table(t));
    }
  }

  CYLON_ASSIGN_OR_RAISE(auto right_tab_combined, right_tab->CombineChunks(memory_pool));

  if (right_tabs.size() > 1) {
    for (const auto &t: right_tabs) {
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(cylon::util::free_table(t));
    }
  }

  return JoinTables(left_tab_combined, right_tab_combined, join_config, joined_table, memory_pool);
}

Status JoinTables(const std::shared_ptr<arrow::Table> &left_tab,
                  const std::shared_ptr<arrow::Table> &right_tab,
                  const config::JoinConfig &join_config,
                  std::shared_ptr<arrow::Table> *joined_table,
                  arrow::MemoryPool *memory_pool) {
  const std::vector<int> &left_indices = join_config.GetLeftColumnIdx();
  const std::vector<int> &right_indices = join_config.GetRightColumnIdx();

  if (left_indices.size() != right_indices.size()) {
    return {Code::Invalid, "left and right index sizes are not equal"};
  }

  if (join_config.GetAlgorithm() == config::HASH) {
    // hash joins
    return HashJoin(left_tab, right_tab, join_config, joined_table, memory_pool);
  } else if (join_config.GetAlgorithm() == config::SORT) {
    // sort joins
    return SortJoin(left_tab, right_tab, join_config, joined_table, memory_pool);
  }
  return {Code::UnknownError, "unknown error"};
}
}  // namespace join
}  // namespace cylon

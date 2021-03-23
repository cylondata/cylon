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
#include "join.hpp"

#include <glog/logging.h>
#include <chrono>
#include <string>

#include "join_utils.hpp"
#include "hash_join.hpp"
#include "sort_join.hpp"
#include "../arrow/arrow_kernels.hpp"
#include "../arrow/arrow_comparator.hpp"

namespace cylon {
namespace join {

arrow::Status JoinTables(const std::vector<std::shared_ptr<arrow::Table>> &left_tabs,
                         const std::vector<std::shared_ptr<arrow::Table>> &right_tabs,
                         const config::JoinConfig &join_config,
                         std::shared_ptr<arrow::Table> *joined_table,
                         arrow::MemoryPool *memory_pool) {
  std::shared_ptr<arrow::Table> left_tab = arrow::ConcatenateTables(left_tabs,
                                                                    arrow::ConcatenateTablesOptions::Defaults(),
                                                                    memory_pool).ValueOrDie();
  std::shared_ptr<arrow::Table> right_tab = arrow::ConcatenateTables(right_tabs,
                                                                     arrow::ConcatenateTablesOptions::Defaults(),
                                                                     memory_pool).ValueOrDie();

  arrow::Result<std::shared_ptr<arrow::Table>> left_combine_res = left_tab->CombineChunks(memory_pool);
  const arrow::Status &left_combine_stat = left_combine_res.status();
  if (!left_combine_stat.ok()) {
    LOG(FATAL) << "Error in combining table chunks of left table." << left_combine_stat.message();
    return left_combine_stat;
  }
  const std::shared_ptr<arrow::Table> &left_tab_combined = left_combine_res.ValueOrDie();

  if (left_tabs.size() > 1) {
    for (const auto &t : left_tabs) {
      arrow::Status status = cylon::util::free_table(t);
      if (!status.ok()) {
        LOG(FATAL) << "Failed to free table" << status.message();
        return status;
      }
    }
  }

  arrow::Result<std::shared_ptr<arrow::Table>> right_combine_res = right_tab->CombineChunks(memory_pool);
  const arrow::Status &right_combine_stat = right_combine_res.status();
  if (!right_combine_stat.ok()) {
    LOG(FATAL) << "Error in combining table chunks of right table." << right_combine_stat.message();
    return right_combine_stat;
  }
  const std::shared_ptr<arrow::Table> &right_tab_combined = right_combine_res.ValueOrDie();

  if (right_tabs.size() > 1) {
    for (const auto &t : right_tabs) {
      arrow::Status status = cylon::util::free_table(t);
      if (!status.ok()) {
        LOG(FATAL) << "Failed to free table" << status.message();
        return status;
      }
    }
  }

  return cylon::join::JoinTables(left_tab_combined, right_tab_combined, join_config, joined_table, memory_pool);
}

arrow::Status JoinTables(const std::shared_ptr<arrow::Table> &left_tab,
                         const std::shared_ptr<arrow::Table> &right_tab,
                         const config::JoinConfig &join_config,
                         std::shared_ptr<arrow::Table> *joined_table,
                         arrow::MemoryPool *memory_pool) {
  const std::vector<int> &left_indices = join_config.GetLeftColumnIdx();
  const std::vector<int> &right_indices = join_config.GetRightColumnIdx();

  if (left_indices.size() != right_indices.size()) {
    return arrow::Status::Invalid("left and right index sizes are not equal");
  }

  if (join_config.GetAlgorithm() == config::HASH) {
    // hash joins
    return HashJoin(left_tab, right_tab, join_config, joined_table, memory_pool);
  } else if (join_config.GetAlgorithm() == config::SORT) {
    // sort joins
    return SortJoin(left_tab, right_tab, join_config, joined_table, memory_pool);
  }
  return arrow::Status::UnknownError("unknown error");
}
}  // namespace join
}  // namespace cylon

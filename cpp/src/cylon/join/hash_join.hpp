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

#ifndef CYLON_CPP_SRC_CYLON_JOIN_HASH_JOIN_HPP_
#define CYLON_CPP_SRC_CYLON_JOIN_HASH_JOIN_HPP_

#include <glog/logging.h>
#include <arrow/api.h>
#include <chrono>

#include "join_config.hpp"
#include "join_utils.hpp"
#include "util/arrow_utils.hpp"
#include "util/macros.hpp"

namespace cylon {
namespace join {

/**
 * Index join of two arrays
 * @param left_idx_col
 * @param right_idx_col
 * @param join_type
 * @param left_table_indices
 * @param right_table_indices
 * @return
 */
arrow::Status ArrayIndexHashJoin(const std::shared_ptr<arrow::Array> &left_idx_col,
                                 const std::shared_ptr<arrow::Array> &right_idx_col,
                                 config::JoinType join_type,
                                 std::vector<int64_t> &left_table_indices,
                                 std::vector<int64_t> &right_table_indices);

/**
 * Performs hash joins on two tables
 * @param ltab
 * @param rtab
 * @param config
 * @param joined_table
 * @param memory_pool
 * @return
 */
arrow::Status HashJoin(const std::shared_ptr<arrow::Table> &ltab,
                       const std::shared_ptr<arrow::Table> &rtab,
                       const config::JoinConfig &config,
                       std::shared_ptr<arrow::Table> *joined_table,
                       arrow::MemoryPool *memory_pool);

}
}

#endif //CYLON_CPP_SRC_CYLON_JOIN_HASH_JOIN_HPP_

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

#ifndef DF78D761_8618_4A9B_AA2C_57ACC7F4E58E
#define DF78D761_8618_4A9B_AA2C_57ACC7F4E58E

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
 * Performs sort joins on two tables
 * @param ltab
 * @param rtab
 * @param config
 * @param joined_table
 * @param memory_pool
 * @return
 */
arrow::Status SortJoin(const std::shared_ptr<arrow::Table> &ltab,
                       const std::shared_ptr<arrow::Table> &rtab,
                       const config::JoinConfig &config,
                       std::shared_ptr<arrow::Table> *joined_table,
                       arrow::MemoryPool *memory_pool);
}
}

#endif /* DF78D761_8618_4A9B_AA2C_57ACC7F4E58E */

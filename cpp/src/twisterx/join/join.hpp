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

#ifndef TWISTERX_TX_JOIN_H
#define TWISTERX_TX_JOIN_H

#include <arrow/table.h>
#include "../arrow/arrow_kernels.hpp"
#include "../arrow/arrow_hash_kernels.hpp"
#include "join_config.h"

namespace twisterx {
namespace join {

arrow::Status joinTables(const std::shared_ptr<arrow::Table> &left_tab,
						 const std::shared_ptr<arrow::Table> &right_tab,
						 twisterx::join::config::JoinConfig join_config,
						 std::shared_ptr<arrow::Table> *joined_table,
						 arrow::MemoryPool *memory_pool = arrow::default_memory_pool());

arrow::Status joinTables(const std::vector<std::shared_ptr<arrow::Table>> &left_tabs,
						 const std::vector<std::shared_ptr<arrow::Table>> &right_tabs,
						 twisterx::join::config::JoinConfig join_config,
						 std::shared_ptr<arrow::Table> *joined_table,
						 arrow::MemoryPool *memory_pool = arrow::default_memory_pool());

}
}
#endif //TWISTERX_TX_JOIN_H

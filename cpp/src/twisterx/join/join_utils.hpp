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

#ifndef CYLON_SRC_JOIN_JOIN_UTILS_HPP_
#define CYLON_SRC_JOIN_JOIN_UTILS_HPP_

#include <arrow/api.h>
#include <map>

namespace cylon {
namespace join {
namespace util {

arrow::Status build_final_table(const std::shared_ptr<std::vector<int64_t>> &left_indices,
                                const std::shared_ptr<std::vector<int64_t>> &right_indices,
                                const std::shared_ptr<arrow::Table> &left_tab,
                                const std::shared_ptr<arrow::Table> &right_tab,
                                std::shared_ptr<arrow::Table> *final_table,
                                arrow::MemoryPool *memory_pool);

arrow::Status CombineChunks(const std::shared_ptr<arrow::Table> &table,
                            int64_t col_index,
                            std::shared_ptr<arrow::Table> &output_table,
                            arrow::MemoryPool *memory_pool);
}
}
}
#endif //CYLON_SRC_JOIN_JOIN_UTILS_HPP_

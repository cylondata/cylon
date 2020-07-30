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

#ifndef CYLON_SRC_UTIL_ARROW_UTILS_HPP_
#define CYLON_SRC_UTIL_ARROW_UTILS_HPP_
#include <arrow/table.h>
#include <arrow/compute/kernel.h>

namespace cylon {
namespace util {

class FunctionContext;

arrow::Status sort_table(std::shared_ptr<arrow::Table> tab, int64_t sort_column_index,
                         std::shared_ptr<arrow::Table> *sorted_table,
                         arrow::MemoryPool *memory_pool = arrow::default_memory_pool());

arrow::Status copy_array_by_indices(const std::shared_ptr<std::vector<int64_t>>& indices,
                                    const std::shared_ptr<arrow::Array>& source_array,
                                    std::shared_ptr<arrow::Array> *copied_array,
                                    arrow::MemoryPool *memory_pool = arrow::default_memory_pool());

/**
 * Free the buffers of a arrow table, after this, the table is no-longer valid
 * @param table the table pointer
 * @return if success
 */
arrow::Status free_table(const std::shared_ptr<arrow::Table> &table);

arrow::Status SortToIndices(arrow::compute::FunctionContext *ctx, const arrow::Array &values,
                            std::shared_ptr<arrow::Array> *offsets);
}  // namespace util
}  // namespace cylon
#endif //CYLON_SRC_UTIL_ARROW_UTILS_HPP_

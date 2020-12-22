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

arrow::Status SortTable(const std::shared_ptr<arrow::Table> &table,
                        int64_t sort_column_index,
                        arrow::MemoryPool *memory_pool,
                        std::shared_ptr<arrow::Table> &sorted_table);

arrow::Status copy_array_by_indices(const std::vector<int64_t> &indices,
                                    const std::shared_ptr<arrow::Array> &source_array,
                                    std::shared_ptr<arrow::Array> *copied_array,
                                    arrow::MemoryPool *memory_pool = arrow::default_memory_pool());

/**
 * Free the buffers of a arrow table, after this, the table is no-longer valid
 * @param table the table pointer
 * @return if success
 */
arrow::Status free_table(const std::shared_ptr<arrow::Table> &table);

/**
 * Create a duplicate of the current array
 */
arrow::Status duplicate(const std::shared_ptr<arrow::ChunkedArray> &cArr,
                        const std::shared_ptr<arrow::Field> &field,
                        arrow::MemoryPool *pool,
                        std::shared_ptr<arrow::ChunkedArray> &out);

arrow::Status SampleTable(std::shared_ptr<arrow::Table> &table,
                          int32_t idx,
                          uint64_t num_samples,
                          std::shared_ptr<arrow::Array> &out);

arrow::Status SampleArray(const std::shared_ptr<arrow::Array> &array,
                          uint64_t num_samples,
                          std::shared_ptr<arrow::Array> &out);

arrow::Status SampleArray(const std::shared_ptr<arrow::ChunkedArray> &array,
                          uint64_t num_samples,
                          std::shared_ptr<arrow::Array> &out);

}  // namespace util
}  // namespace cylon
#endif //CYLON_SRC_UTIL_ARROW_UTILS_HPP_

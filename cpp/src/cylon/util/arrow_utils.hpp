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

#include <arrow/compute/kernel.h>
#include <arrow/table.h>

namespace cylon {
namespace util {

/**
 * returns the sign bit of a number type member. returns (v<0)? 0: 1
 * ref: https://graphics.stanford.edu/~seander/bithacks.html#CopyIntegerSign
 * @tparam T
 * @param v
 * @return
 */
template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>>
static inline constexpr bool GetSignBit(const T v) {
  return 1 ^ (static_cast<typename std::make_unsigned<T>::type>(v) >> (sizeof(T) * CHAR_BIT - 1));
}

/**
 * take absolute value of integral types
 * ref: https://graphics.stanford.edu/~seander/bithacks.html#IntegerAbs
 */
template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>>
static inline constexpr T GetAbs(const T v) {
  const int mask = v >> (sizeof(T) * CHAR_BIT - 1);
  return (v ^ mask) - mask;
}

/**
 * ref: https://stackoverflow.com/questions/47981/how-do-you-set-clear-and-toggle-a-single-bit
 * @tparam T
 * @tparam BIT
 * @param v
 * @return
 */
static inline constexpr int64_t SetBit(const int64_t v) {
  return v | 1UL << (sizeof(int64_t) * CHAR_BIT - 1);
}
static inline constexpr int64_t ClearBit(const int64_t v) {
  return v & ~(1UL << (sizeof(int64_t) * CHAR_BIT - 1));
}
static inline constexpr size_t CheckBit(const int64_t v) {
  return (v >> (sizeof(int64_t) * CHAR_BIT - 1)) & 1U;
}

arrow::Status SortTable(const std::shared_ptr<arrow::Table> &table, int64_t sort_column_index,
                        arrow::MemoryPool *memory_pool, std::shared_ptr<arrow::Table> &sorted_table,
                        bool ascending = true);

arrow::Status SortTableMultiColumns(const std::shared_ptr<arrow::Table> &table,
                                    const std::vector<int32_t> &sort_column_indices,
                                    arrow::MemoryPool *memory_pool,
                                    std::shared_ptr<arrow::Table> &sorted_table,
                                    const std::vector<bool> &sort_column_directions);

arrow::Status SortTableMultiColumns(const std::shared_ptr<arrow::Table> &table,
                                    const std::vector<int32_t> &sort_column_indices,
                                    arrow::MemoryPool *memory_pool,
                                    std::shared_ptr<arrow::Table> &sorted_table,
                                    const std::vector<bool> &sort_column_directions);

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
                        const std::shared_ptr<arrow::Field> &field, arrow::MemoryPool *pool,
                        std::shared_ptr<arrow::ChunkedArray> &out);

arrow::Status SampleTable(std::shared_ptr<arrow::Table> &table, int32_t idx, uint64_t num_samples,
                          std::shared_ptr<arrow::Array> &out);

arrow::Status SampleArray(const std::shared_ptr<arrow::Array> &array, uint64_t num_samples,
                          std::shared_ptr<arrow::Array> &out);

arrow::Status SampleArray(const std::shared_ptr<arrow::ChunkedArray> &array, uint64_t num_samples,
                          std::shared_ptr<arrow::Array> &out);

std::shared_ptr<arrow::Array> GetChunkOrEmptyArray(const std::shared_ptr<arrow::ChunkedArray> &column, int chunk);

}  // namespace util
}  // namespace cylon
#endif  // CYLON_SRC_UTIL_ARROW_UTILS_HPP_

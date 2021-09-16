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

#include <arrow/api.h>
#include <arrow/table.h>
#include <arrow/visitor_inline.h>

#include "cylon/status.hpp"
#include "cylon/util/macros.hpp"

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
  return v | int64_t(1) << (sizeof(int64_t) * CHAR_BIT - 1);
}
static inline constexpr int64_t ClearBit(const int64_t v) {
  return v & ~(int64_t(1) << (sizeof(int64_t) * CHAR_BIT - 1));
}
static inline constexpr int64_t CheckBit(const int64_t v) {
  return (v >> (sizeof(int64_t) * CHAR_BIT - 1)) & int64_t(1);
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
arrow::Status Duplicate(const std::shared_ptr<arrow::ChunkedArray> &cArr, arrow::MemoryPool *pool,
                        std::shared_ptr<arrow::ChunkedArray> &out);
arrow::Status Duplicate(const std::shared_ptr<arrow::Table> &table, arrow::MemoryPool *pool,
                        std::shared_ptr<arrow::Table> &out);

/**
 * Sample array
 * @param array
 * @param num_samples
 * @param out
 * @param pool
 * @return
 */
arrow::Status SampleArray(const std::shared_ptr<arrow::ChunkedArray> &array, uint64_t num_samples,
                          std::shared_ptr<arrow::Array> &out,
                          arrow::MemoryPool *pool = arrow::default_memory_pool());

arrow::Status SampleArray(std::shared_ptr<arrow::Table> &table, int32_t idx, uint64_t num_samples,
                          std::shared_ptr<arrow::Array> &out,
                          arrow::MemoryPool *pool = arrow::default_memory_pool());

arrow::Status SampleArray(const std::shared_ptr<arrow::Array> &array, uint64_t num_samples,
                          std::shared_ptr<arrow::Array> &out,
                          arrow::MemoryPool *pool = arrow::default_memory_pool());

std::shared_ptr<arrow::Array> GetChunkOrEmptyArray(const std::shared_ptr<arrow::ChunkedArray> &column, int chunk,
                                                   arrow::MemoryPool *pool = arrow::default_memory_pool());

arrow::Status GetConcatenatedColumn(const std::shared_ptr<arrow::Table> &table, int col_id,
                                    std::shared_ptr<arrow::Array> *output,
                                    arrow::MemoryPool *pool = arrow::default_memory_pool());

inline bool IsMutable(const std::shared_ptr<arrow::Array> &array) {
  for (auto &&buff: array->data()->buffers) {
    if (buff != nullptr && !buff->is_mutable()) return false;
  }
  return true;
}

/**
 * Return the number of splits according to cache size
 * @param values
 * @return
 */
uint64_t GetNumberSplitsToFitInCache(int64_t total_bytes, int total_elements, int parallel);

/**
 * Return the number of bytes and elements in this table in the given columns
 * @param table
 * @param columns
 * @return
 */
std::array<int64_t, 2> GetBytesAndElements(std::shared_ptr<arrow::Table> table,
                                           const std::vector<int> &columns);

/**
 * Creates an empty array of type `data_type`
 * @param data_type
 * @param output
 * @param pool
 * @return
 */
arrow::Status CreateEmptyArray(const std::shared_ptr<arrow::DataType> &data_type,
                               std::shared_ptr<arrow::Array> *output,
                               arrow::MemoryPool *pool = arrow::default_memory_pool());

/**
 * Check an array for uniqueness
 * @param array
 * @param pool
 * @return
 */
arrow::Status IsUnique(const std::shared_ptr<arrow::Array> &array, bool *output,
                       arrow::MemoryPool *pool = arrow::default_memory_pool());

template<typename ArrowT, typename Enable = void>
struct ArrowScalarValue {};

template<typename ArrowT>
struct ArrowScalarValue<ArrowT, arrow::enable_if_has_c_type<ArrowT>> {
  using ScalarT = typename arrow::TypeTraits<ArrowT>::ScalarType;
  using ValueT = typename ArrowT::c_type;

  static ValueT Extract(const std::shared_ptr<arrow::Scalar> &scalar) {
    return std::static_pointer_cast<ScalarT>(scalar)->value;
  }
};

template<typename ArrowT>
struct ArrowScalarValue<ArrowT, arrow::enable_if_has_string_view<ArrowT>> {
  using ScalarT = typename arrow::TypeTraits<ArrowT>::ScalarType;
  using ValueT = arrow::util::string_view;

  static ValueT Extract(const std::shared_ptr<arrow::Scalar> &scalar) {
    return ValueT(*(std::static_pointer_cast<ScalarT>(scalar))->value);
  }
};

/**
 * Find indices of `search_param` value. If not found, IndexError Status will be returned.
 *
 * @param array
 * @param search_param
 * @param locations
 * @return arrow::Status
 */
template<typename ArrowT>
arrow::Status FindIndices(const std::shared_ptr<arrow::Array> &index_array_,
                          const std::shared_ptr<arrow::Scalar> &search_param,
                          std::shared_ptr<arrow::Int64Array> *locations,
                          arrow::MemoryPool *pool) {
  using ValueT = typename ArrowScalarValue<ArrowT>::ValueT;

  // if search param is null and index_array doesn't have any nulls, return empty array
  if (!search_param->is_valid && index_array_->null_count() == 0) {
    return arrow::Status::KeyError("Key not found");
  }

  // reserve conservatively
  arrow::Int64Builder builder(pool);
  RETURN_ARROW_STATUS_IF_FAILED(builder.Reserve(index_array_->length()));

  int64_t idx = 0;
  const auto &arr_data = *index_array_->data();
  if (search_param->is_valid) {
    // param is valid, so search only on the valid elements
    // search param needs to be casted here to support castable params from python
    auto res = search_param->CastTo(index_array_->type());
    RETURN_ARROW_STATUS_IF_FAILED(res.status());
    const ValueT &cast_val = ArrowScalarValue<ArrowT>::Extract(res.ValueOrDie());

    arrow::VisitArrayDataInline<ArrowT>(
        arr_data,
        [&](ValueT val) {
          if (cast_val == val) {
            builder.UnsafeAppend(idx);
          }
          idx++;
        },
        [&]() {  // nothing to do for nulls
          idx++;
        });
  } else {
    // param is null. search only on the null elements. So, search in the validity bitmap of array
    const auto &val_buff = arr_data.buffers[0];
    arrow::VisitNullBitmapInline(
        val_buff->data(), arr_data.offset, arr_data.length, 0,
        [&]() {  // nothing to do for valid-values
          idx++;
        },
        [&]() {
          builder.UnsafeAppend(idx);
          idx++;
        });
  }

  if (builder.length() == 0) {
    return arrow::Status::KeyError("Key not found");
  }

  return builder.Finish(locations);
}

/**
 * @brief Find indices of the values in `search_param` array. If not found, IndexError Status will be returned.
 *
 * @param array
 * @param search_param
 * @param locations
 * @return arrow::Status
 */
arrow::Status FindIndices(const std::shared_ptr<arrow::Array> &array,
                          const std::shared_ptr<arrow::Array> &search_param,
                          std::shared_ptr<arrow::Int64Array> *locations,
                          arrow::MemoryPool *pool = arrow::default_memory_pool());

/**
 * Find index of `search_param` value. If not found, IndexError Status will be returned.
 *
 * @param array
 * @param search_param
 * @param locations
 * @return arrow::Status
 */
template<typename ArrowT>
arrow::Status FindIndex(const std::shared_ptr<arrow::Array> &index_array_,
                        const std::shared_ptr<arrow::Scalar> &search_param, int64_t *index) {
  using ValueT = typename ArrowScalarValue<ArrowT>::ValueT;

  if (!search_param->is_valid && index_array_->null_count() == 0) {
    return arrow::Status::KeyError("Key not found");
  }

  int64_t idx = 0;
  const auto &arr_data = *index_array_->data();
  if (search_param->is_valid) {
    // param is valid, so search only on the valid elements

    // search param needs to be casted here to support castable params from python
    const auto &res = search_param->CastTo(index_array_->type());
    RETURN_ARROW_STATUS_IF_FAILED(res.status());
    const ValueT &cast_val = ArrowScalarValue<ArrowT>::Extract(res.ValueOrDie());

    arrow::VisitArrayDataInline<ArrowT>(
        arr_data,
        [&](ValueT val) {
          if (cast_val == val) {
            *index = idx;
            return arrow::Status::Cancelled("");  // this will break the visit loop
          }
          idx++;
          return arrow::Status::OK();
        },
        [&]() {  // nothing to do for nulls
          idx++;
          return arrow::Status::OK();
        });
  } else {
    // param is null. search only on the null elements. So, search in the validity bitmap of array
    const auto &val_buff = arr_data.buffers[0];
    arrow::VisitNullBitmapInline(
        val_buff->data(), arr_data.offset, arr_data.length, 0,
        [&]() {  // nothing to do for valid-values
          idx++;
          return arrow::Status::OK();
        },
        [&]() {
          *index = idx;
          return arrow::Status::Cancelled("");  // this will break the visit loop
        });
  }

  // if the index has reached the end, that means the param is not found
  if (idx == index_array_->length()) {
    return arrow::Status::KeyError("Key not found");
  }

  return arrow::Status::OK();
}

}  // namespace util
}  // namespace cylon
#endif  // CYLON_SRC_UTIL_ARROW_UTILS_HPP_

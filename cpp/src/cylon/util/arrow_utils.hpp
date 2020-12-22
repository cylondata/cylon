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

template<typename ITER, typename TYPE>
arrow::Status do_copy_numeric_array(const ITER &indices_begin,
                                    const ITER &indices_end,
                                    int64_t size,
                                    const std::shared_ptr<arrow::Array> &data_array,
                                    std::shared_ptr<arrow::Array> &copied_array,
                                    arrow::MemoryPool *memory_pool) {
  arrow::NumericBuilder<TYPE> array_builder(memory_pool);
  arrow::Status status = array_builder.Reserve(size);
  if (!status.ok()) {
//    LOG(FATAL) << "Failed to reserve memory when re arranging the array based on indices. " << status.ToString();
    return status;
  }

  auto casted_array = std::static_pointer_cast<arrow::NumericArray<TYPE>>(data_array);
  for (auto index = indices_begin; index != indices_end; index++) {
    // handle -1 index : comes in left, right joins
    if (*index == -1) {
      array_builder.UnsafeAppendNull();
      continue;
    }

    if (casted_array->length() <= *index) {
      return arrow::Status::Invalid(
          "INVALID INDEX " + std::to_string(*index) + " LENGTH " + std::to_string(casted_array->length()));
    }
    array_builder.UnsafeAppend(casted_array->Value(*index));
  }
  return array_builder.Finish(&copied_array);
}

template<typename ITER>
arrow::Status copy_array_by_indices(const ITER &indices_begin,
                                    const ITER &indices_end,
                                    int64_t size,
                                    const std::shared_ptr<arrow::Array> &data_array,
                                    std::shared_ptr<arrow::Array> &copied_array,
                                    arrow::MemoryPool *memory_pool = arrow::default_memory_pool()) {
  switch (data_array->type_id()) {
    case arrow::Type::BOOL:
      return do_copy_numeric_array<ITER, arrow::BooleanType>(indices_begin, indices_end, size,
                                                             data_array,
                                                             copied_array,
                                                             memory_pool);
    case arrow::Type::UINT8:
      return do_copy_numeric_array<ITER, arrow::UInt8Type>(indices_begin, indices_end, size,
                                                           data_array,
                                                           copied_array,
                                                           memory_pool);
    case arrow::Type::INT8:
      return do_copy_numeric_array<ITER, arrow::Int8Type>(indices_begin, indices_end, size,
                                                          data_array,
                                                          copied_array,
                                                          memory_pool);
    case arrow::Type::UINT16:
      return do_copy_numeric_array<ITER, arrow::UInt16Type>(indices_begin, indices_end, size,
                                                            data_array,
                                                            copied_array,
                                                            memory_pool);
    case arrow::Type::INT16:
      return do_copy_numeric_array<ITER, arrow::Int16Type>(indices_begin, indices_end, size,
                                                           data_array,
                                                           copied_array,
                                                           memory_pool);
    case arrow::Type::UINT32:
      return do_copy_numeric_array<ITER, arrow::UInt32Type>(indices_begin, indices_end, size,
                                                            data_array,
                                                            copied_array,
                                                            memory_pool);
    case arrow::Type::INT32:
      return do_copy_numeric_array<ITER, arrow::Int32Type>(indices_begin, indices_end, size,
                                                           data_array,
                                                           copied_array,
                                                           memory_pool);
    case arrow::Type::UINT64:
      return do_copy_numeric_array<ITER, arrow::UInt64Type>(indices_begin, indices_end, size,
                                                            data_array,
                                                            copied_array,
                                                            memory_pool);
    case arrow::Type::INT64:
      return do_copy_numeric_array<ITER, arrow::Int64Type>(indices_begin, indices_end, size,
                                                           data_array,
                                                           copied_array,
                                                           memory_pool);
    case arrow::Type::HALF_FLOAT:
      return do_copy_numeric_array<ITER, arrow::HalfFloatType>(indices_begin, indices_end, size,
                                                               data_array,
                                                               copied_array,
                                                               memory_pool);
    case arrow::Type::FLOAT:
      return do_copy_numeric_array<ITER, arrow::FloatType>(indices_begin, indices_end, size,
                                                           data_array,
                                                           copied_array,
                                                           memory_pool);
    case arrow::Type::DOUBLE:
      return do_copy_numeric_array<ITER, arrow::DoubleType>(indices_begin, indices_end, size,
                                                            data_array,
                                                            copied_array,
                                                            memory_pool);

    default:return arrow::Status::Invalid("Un-supported type");
  }
}

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

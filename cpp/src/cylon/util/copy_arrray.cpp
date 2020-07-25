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

#include <arrow/compute/api.h>
#include <arrow/api.h>
#include <glog/logging.h>
#include "arrow_utils.hpp"
#include "../status.hpp"

namespace cylon {
namespace util {

template<typename TYPE>
arrow::Status do_copy_numeric_array(const std::shared_ptr<std::vector<int64_t>> &indices,
                                    std::shared_ptr<arrow::Array> data_array,
                                    std::shared_ptr<arrow::Array> *copied_array,
                                    arrow::MemoryPool *memory_pool) {
  arrow::NumericBuilder<TYPE> array_builder(memory_pool);
  arrow::Status status = array_builder.Reserve(indices->size());
  if (status != arrow::Status::OK()) {
    LOG(FATAL) << "Failed to reserve memory when re arranging the array based on indices. "
               << status.ToString();
    return status;
  }

  auto casted_array = std::static_pointer_cast<arrow::NumericArray<TYPE>>(data_array);
  for (auto &index : *indices) {
    // handle -1 index : comes in left, right joins
    if (index == -1) {
      array_builder.UnsafeAppendNull();
      continue;
    }

    if (casted_array->length() <= index) {
      LOG(FATAL) << "INVALID INDEX " << index << " LENGTH " << casted_array->length();
    }
    array_builder.UnsafeAppend(casted_array->Value(index));
  }
  return array_builder.Finish(copied_array);
}

arrow::Status do_copy_binary_array(std::shared_ptr<std::vector<int64_t>> indices,
                                   std::shared_ptr<arrow::Array> data_array,
                                   std::shared_ptr<arrow::Array> *copied_array,
                                   arrow::MemoryPool *memory_pool) {
  arrow::BinaryBuilder binary_builder(memory_pool);
  auto casted_array = std::static_pointer_cast<arrow::BinaryArray>(data_array);
  for (auto &index : *indices) {
    if (casted_array->length() <= index) {
      LOG(FATAL) << "INVALID INDEX " << index << " LENGTH " << casted_array->length();
    }
    int32_t out;
    const uint8_t *data = casted_array->GetValue(index, &out);
    auto status = binary_builder.Append(data, out);
    if (status != arrow::Status::OK()) {
      LOG(FATAL) << "Failed to append rearranged data points to the array builder. "
                 << status.ToString();
      return status;
    }
  }
  return binary_builder.Finish(copied_array);
}

arrow::Status do_copy_fixed_binary_array(std::shared_ptr<std::vector<int64_t>> indices,
                                         std::shared_ptr<arrow::Array> data_array,
                                         std::shared_ptr<arrow::Array> *copied_array,
                                         arrow::MemoryPool *memory_pool) {
  arrow::FixedSizeBinaryBuilder binary_builder(data_array->type(), memory_pool);
  auto casted_array = std::static_pointer_cast<arrow::FixedSizeBinaryArray>(data_array);
  for (auto &index : *indices) {
    if (casted_array->length() <= index) {
      LOG(FATAL) << "INVALID INDEX " << index << " LENGTH " << casted_array->length();
    }
    const uint8_t *data = casted_array->GetValue(index);
    arrow::Status status = binary_builder.Append(data);
    if (status != arrow::Status::OK()) {
      LOG(FATAL) << "Failed to append rearranged data points to the array builder. "
                 << status.ToString();
      return status;
    }
  }
  return binary_builder.Finish(copied_array);
}

template<typename TYPE>
arrow::Status do_copy_numeric_list(std::shared_ptr<std::vector<int64_t>> indices,
                                   std::shared_ptr<arrow::Array> data_array,
                                   std::shared_ptr<arrow::Array> *copied_array,
                                   arrow::MemoryPool *memory_pool) {
  arrow::ListBuilder list_builder(memory_pool,
      std::make_shared<arrow::NumericBuilder<TYPE>>(memory_pool));
  arrow::NumericBuilder<TYPE> &value_builder =
      *(static_cast<arrow::NumericBuilder<TYPE> *>(list_builder.value_builder()));
  auto casted_array = std::static_pointer_cast<arrow::ListArray>(data_array);
  for (auto &index : *indices) {
    arrow::Status status = list_builder.Append();
    if (status != arrow::Status::OK()) {
      LOG(FATAL) << "Failed to append rearranged data points to the array builder. "
                 << status.ToString();
      return status;
    }
    auto numericArray = std::static_pointer_cast<arrow::NumericArray<TYPE>>(
        casted_array->Slice(index));

    for (int64_t n = 0; n < numericArray->length(); n++) {
      status = value_builder.Append(numericArray->Value(n));
      if (status != arrow::Status::OK()) {
        LOG(FATAL) << "Failed to append rearranged data points to the array builder. "
                   << status.ToString();
        return status;
      }
    }
  }
  return list_builder.Finish(copied_array);
}

arrow::Status copy_array_by_indices(const std::shared_ptr<std::vector<int64_t>> &indices,
                                    const std::shared_ptr<arrow::Array> &data_array,
                                    std::shared_ptr<arrow::Array> *copied_array,
                                    arrow::MemoryPool *memory_pool) {
  switch (data_array->type()->id()) {
    case arrow::Type::UINT8:
      return do_copy_numeric_array<arrow::UInt8Type>(indices,
                                                     data_array,
                                                     copied_array,
                                                     memory_pool);
    case arrow::Type::INT8:
      return do_copy_numeric_array<arrow::Int8Type>(indices,
                                                    data_array,
                                                    copied_array,
                                                    memory_pool);
    case arrow::Type::UINT16:
      return do_copy_numeric_array<arrow::Int16Type>(indices,
                                                     data_array,
                                                     copied_array,
                                                     memory_pool);
    case arrow::Type::INT16:
      return do_copy_numeric_array<arrow::Int16Type>(indices,
                                                     data_array,
                                                     copied_array,
                                                     memory_pool);
    case arrow::Type::UINT32:
      return do_copy_numeric_array<arrow::UInt32Type>(indices,
                                                      data_array,
                                                      copied_array,
                                                      memory_pool);
    case arrow::Type::INT32:
      return do_copy_numeric_array<arrow::Int32Type>(indices,
                                                     data_array,
                                                     copied_array,
                                                     memory_pool);
    case arrow::Type::UINT64:
      return do_copy_numeric_array<arrow::UInt64Type>(indices,
                                                      data_array,
                                                      copied_array,
                                                      memory_pool);
    case arrow::Type::INT64:
      return do_copy_numeric_array<arrow::Int64Type>(indices,
                                                     data_array,
                                                     copied_array,
                                                     memory_pool);
    case arrow::Type::HALF_FLOAT:
      return do_copy_numeric_array<arrow::HalfFloatType>(indices,
                                                         data_array,
                                                         copied_array,
                                                         memory_pool);
    case arrow::Type::FLOAT:
      return do_copy_numeric_array<arrow::FloatType>(indices,
                                                     data_array,
                                                     copied_array,
                                                     memory_pool);
    case arrow::Type::DOUBLE:
      return do_copy_numeric_array<arrow::DoubleType>(indices,
                                                      data_array,
                                                      copied_array,
                                                      memory_pool);
    case arrow::Type::STRING:
      return do_copy_binary_array(indices, data_array,
          copied_array, memory_pool);
    case arrow::Type::BINARY:
      return do_copy_binary_array(indices, data_array,
          copied_array, memory_pool);
    case arrow::Type::FIXED_SIZE_BINARY:
      return do_copy_fixed_binary_array(indices,
                                        data_array,
                                        copied_array,
                                        memory_pool);
    case arrow::Type::LIST: {
      auto t_value = std::static_pointer_cast<arrow::ListType>(data_array->type());
      switch (t_value->value_type()->id()) {
        case arrow::Type::UINT8:
          return do_copy_numeric_list<arrow::UInt8Type>(indices,
                                                        data_array,
                                                        copied_array,
                                                        memory_pool);
        case arrow::Type::INT8:
          return do_copy_numeric_list<arrow::Int8Type>(indices,
                                                       data_array,
                                                       copied_array,
                                                       memory_pool);
        case arrow::Type::UINT16:
          return do_copy_numeric_list<arrow::Int16Type>(indices,
                                                        data_array,
                                                        copied_array,
                                                        memory_pool);
        case arrow::Type::INT16:
          return do_copy_numeric_list<arrow::Int16Type>(indices,
                                                        data_array,
                                                        copied_array,
                                                        memory_pool);
        case arrow::Type::UINT32:
          return do_copy_numeric_list<arrow::UInt32Type>(indices,
                                                         data_array,
                                                         copied_array,
                                                         memory_pool);
        case arrow::Type::INT32:
          return do_copy_numeric_list<arrow::Int32Type>(indices,
                                                        data_array,
                                                        copied_array,
                                                        memory_pool);
        case arrow::Type::UINT64:
          return do_copy_numeric_list<arrow::UInt64Type>(indices,
                                                         data_array,
                                                         copied_array,
                                                         memory_pool);
        case arrow::Type::INT64:
          return do_copy_numeric_list<arrow::Int64Type>(indices,
                                                        data_array,
                                                        copied_array,
                                                        memory_pool);
        case arrow::Type::HALF_FLOAT:
          return do_copy_numeric_list<arrow::HalfFloatType>(indices,
                                                            data_array,
                                                            copied_array,
                                                            memory_pool);
        case arrow::Type::FLOAT:
          return do_copy_numeric_list<arrow::FloatType>(indices,
                                                        data_array,
                                                        copied_array,
                                                        memory_pool);
        case arrow::Type::DOUBLE:
          return do_copy_numeric_list<arrow::DoubleType>(indices,
                                                         data_array,
                                                         copied_array,
                                                         memory_pool);
        default:
          return arrow::Status::Invalid("Un-supported type");
      }
    }
    default:
      return arrow::Status::Invalid("Un-supported type");
  }
}

}  // namespace util
}  // namespace cylon

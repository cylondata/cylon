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

#ifndef CYLON_ARROW_KERNELS_H
#define CYLON_ARROW_KERNELS_H

#include <arrow/api.h>
#include <arrow/compute/kernel.h>
#include <glog/logging.h>
#include <iostream>
#include "../status.hpp"
#include "util/sort.hpp"
#include "util/macros.hpp"

namespace cylon {

class ArrowArraySplitKernel {
 public:
  explicit ArrowArraySplitKernel(arrow::MemoryPool *pool) : pool_(pool) {}

  /**
   * Merge the values in the column and return an array
   * @param ctx
   * @param values
   * @param targets
   * @param out_length
   * @param out
   * @return
   */
  virtual Status Split(const std::shared_ptr<arrow::ChunkedArray> &values,
                       uint32_t num_partitions,
                       const std::vector<uint32_t> &target_partitions,
                       const std::vector<uint32_t> &counts,
                       std::vector<std::shared_ptr<arrow::Array>> &output) = 0;

 protected:
  arrow::MemoryPool *pool_;
};

template<typename TYPE, typename = typename std::enable_if<
    arrow::is_number_type<TYPE>::value | arrow::is_boolean_type<TYPE>::value>::type>
class ArrowArrayNumericSplitKernel : public ArrowArraySplitKernel {
 public:
  explicit ArrowArrayNumericSplitKernel(arrow::MemoryPool *pool) : ArrowArraySplitKernel(pool) {}

  using ARROW_ARRAY_T = typename arrow::TypeTraits<TYPE>::ArrayType;
  using ARROW_BUILDER_T = typename arrow::TypeTraits<TYPE>::BuilderType;

  Status Split(const std::shared_ptr<arrow::ChunkedArray> &values,
               uint32_t num_partitions,
               const std::vector<uint32_t> &target_partitions,
               const std::vector<uint32_t> &counts,
               std::vector<std::shared_ptr<arrow::Array>> &output) override {

    if ((size_t) values->length() != target_partitions.size()) {
      return Status(Code::ExecutionError, "values rows != target_partitions length");
    }

    std::vector<std::unique_ptr<ARROW_BUILDER_T>> builders;
    builders.reserve(num_partitions);
    for (uint32_t i = 0; i < num_partitions; i++) {
      builders.emplace_back(new ARROW_BUILDER_T(pool_));
      const auto &status = builders.back()->Reserve(counts[i]);
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(status)
    }

    size_t offset = 0;
    for (const auto &array:values->chunks()) {
      std::shared_ptr<ARROW_ARRAY_T> casted_array = std::static_pointer_cast<ARROW_ARRAY_T>(array);
      const int64_t arr_len = array->length();
      for (int64_t i = 0; i < arr_len; i++, offset++) {
        builders[target_partitions[offset]]->UnsafeAppend(casted_array->Value(i));
      }
    }

    output.reserve(num_partitions);
    for (uint32_t i = 0; i < num_partitions; i++) {
      std::shared_ptr<arrow::Array> array;
      const auto &status = builders[i]->Finish(&array);
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(status)
      output.push_back(array);
    }

    return Status::OK();
  }
};

class FixedBinaryArraySplitKernel : public ArrowArraySplitKernel {
 public:
  using ARROW_ARRAY_T = arrow::FixedSizeBinaryArray;
  using ARROW_BUILDER_T = arrow::FixedSizeBinaryBuilder;

  explicit FixedBinaryArraySplitKernel(arrow::MemoryPool *pool) : ArrowArraySplitKernel(pool) {}

  Status Split(const std::shared_ptr<arrow::ChunkedArray> &values,
               uint32_t num_partitions,
               const std::vector<uint32_t> &target_partitions,
               const std::vector<uint32_t> &counts,
               std::vector<std::shared_ptr<arrow::Array>> &output) override;
};

template<typename TYPE>
class BinaryArraySplitKernel : public ArrowArraySplitKernel {
 public:
  using ARROW_ARRAY_T = typename arrow::TypeTraits<TYPE>::ArrayType;
  using ARROW_BUILDER_T = typename arrow::TypeTraits<TYPE>::BuilderType;
  using ARROW_OFFSET_T = typename TYPE::offset_type;

  explicit BinaryArraySplitKernel(arrow::MemoryPool *pool) : ArrowArraySplitKernel(pool) {
  }

  Status Split(const std::shared_ptr<arrow::ChunkedArray> &values,
               uint32_t num_partitions,
               const std::vector<uint32_t> &target_partitions,
               const std::vector<uint32_t> &counts,
               std::vector<std::shared_ptr<arrow::Array>> &output) override {

    if ((size_t) values->length() != target_partitions.size()) {
      return Status(Code::ExecutionError, "values rows != target_partitions length");
    }

    std::vector<std::unique_ptr<ARROW_BUILDER_T>> builders;
    builders.reserve(num_partitions);
    for (uint32_t i = 0; i < num_partitions; i++) {
      builders.emplace_back(new ARROW_BUILDER_T(pool_));
    }

    size_t offset = 0;
    for (const auto &array:values->chunks()) {
      std::shared_ptr<ARROW_ARRAY_T> casted_array = std::static_pointer_cast<ARROW_ARRAY_T>(array);
      const int64_t arr_len = array->length();
      for (int64_t i = 0; i < arr_len; i++, offset++) {
        ARROW_OFFSET_T length = 0;
        const uint8_t *value = casted_array->GetValue(i, &length);
        const auto &a_status = builders[target_partitions[offset]]->Append(value, length);
        RETURN_CYLON_STATUS_IF_ARROW_FAILED(a_status)
      }
    }

    output.reserve(num_partitions);
    for (uint32_t i = 0; i < num_partitions; i++) {
      std::shared_ptr<arrow::Array> array;
      const auto &status = builders[i]->Finish(&array);
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(status)
      output.push_back(array);
    }

    return Status::OK();
  }
};

using UInt8ArraySplitter = ArrowArrayNumericSplitKernel<arrow::UInt8Type>;
using UInt16ArraySplitter = ArrowArrayNumericSplitKernel<arrow::UInt16Type>;
using UInt32ArraySplitter = ArrowArrayNumericSplitKernel<arrow::UInt32Type>;
using UInt64ArraySplitter = ArrowArrayNumericSplitKernel<arrow::UInt64Type>;

using Int8ArraySplitter = ArrowArrayNumericSplitKernel<arrow::Int8Type>;
using Int16ArraySplitter = ArrowArrayNumericSplitKernel<arrow::Int16Type>;
using Int32ArraySplitter = ArrowArrayNumericSplitKernel<arrow::Int32Type>;
using Int64ArraySplitter = ArrowArrayNumericSplitKernel<arrow::Int64Type>;

using HalfFloatArraySplitter = ArrowArrayNumericSplitKernel<arrow::HalfFloatType>;
using FloatArraySplitter = ArrowArrayNumericSplitKernel<arrow::FloatType>;
using DoubleArraySplitter = ArrowArrayNumericSplitKernel<arrow::DoubleType>;

std::unique_ptr<ArrowArraySplitKernel> CreateSplitter(const std::shared_ptr<arrow::DataType> &type,
                                                      arrow::MemoryPool *pool);

class IndexSortKernel {
 public:
  explicit IndexSortKernel(arrow::MemoryPool *pool) : pool_(pool) {}

  /**
   * Sort the values in the column and return an array with the indices
   * @param ctx
   * @param values
   * @param targets
   * @param out_length
   * @param out
   * @return
   */
  virtual arrow::Status Sort(std::shared_ptr<arrow::Array> &values, std::shared_ptr<arrow::Array> &out) = 0;

 protected:
  arrow::MemoryPool *pool_;
};

template<typename TYPE, typename = typename std::enable_if<
    arrow::is_number_type<TYPE>::value | arrow::is_boolean_type<TYPE>::value>::type>
class NumericIndexSortKernel : public IndexSortKernel {
 public:
  using T = typename TYPE::c_type;

  explicit NumericIndexSortKernel(arrow::MemoryPool *pool) : IndexSortKernel(pool) {}

  arrow::Status Sort(std::shared_ptr<arrow::Array> &values, std::shared_ptr<arrow::Array> &offsets) override {
    auto array = std::static_pointer_cast<arrow::NumericArray<TYPE>>(values);
    const T *left_data = array->raw_values();
    int64_t buf_size = values->length() * sizeof(uint64_t);

    arrow::Result<std::unique_ptr<arrow::Buffer>> result = AllocateBuffer(buf_size + 1, pool_);
    const arrow::Status &status = result.status();
    if (!status.ok()) {
      LOG(FATAL) << "Failed to allocate sort indices - " << status.message();
      return status;
    }
    std::shared_ptr<arrow::Buffer> indices_buf = std::move(result.ValueOrDie());

    auto *indices_begin = reinterpret_cast<int64_t *>(indices_buf->mutable_data());
    for (int64_t i = 0; i < values->length(); i++) {
      indices_begin[i] = i;
    }
    int64_t *indices_end = indices_begin + values->length();
    std::sort(indices_begin, indices_end, [left_data](uint64_t left, uint64_t right) {
      return left_data[left] < left_data[right];
    });
    offsets = std::make_shared<arrow::UInt64Array>(values->length(), indices_buf);
    return arrow::Status::OK();
  }
};

using UInt8ArraySorter = NumericIndexSortKernel<arrow::UInt8Type>;
using UInt16ArraySorter = NumericIndexSortKernel<arrow::UInt16Type>;
using UInt32ArraySorter = NumericIndexSortKernel<arrow::UInt32Type>;
using UInt64ArraySorter = NumericIndexSortKernel<arrow::UInt64Type>;
using Int8ArraySorter = NumericIndexSortKernel<arrow::Int8Type>;
using Int16ArraySorter = NumericIndexSortKernel<arrow::Int16Type>;
using Int32ArraySorter = NumericIndexSortKernel<arrow::Int32Type>;
using Int64ArraySorter = NumericIndexSortKernel<arrow::Int64Type>;
using HalfFloatArraySorter = NumericIndexSortKernel<arrow::HalfFloatType>;
using FloatArraySorter = NumericIndexSortKernel<arrow::FloatType>;
using DoubleArraySorter = NumericIndexSortKernel<arrow::DoubleType>;

arrow::Status SortIndices(arrow::MemoryPool *memory_pool, std::shared_ptr<arrow::Array> &values,
                          std::shared_ptr<arrow::Array> &offsets);

class InplaceIndexSortKernel {
 public:
  explicit InplaceIndexSortKernel(arrow::MemoryPool *pool) : pool_(pool) {}

  /**
   * Sort the values in the column and return an array with the indices
   * @param ctx
   * @param values
   * @param targets
   * @param out_length
   * @param out
   * @return
   */
  virtual arrow::Status Sort(std::shared_ptr<arrow::Array> &values, std::shared_ptr<arrow::UInt64Array> &out) = 0;
 protected:
  arrow::MemoryPool *pool_;
};

template<typename TYPE, typename = typename std::enable_if<
    arrow::is_number_type<TYPE>::value | arrow::is_boolean_type<TYPE>::value>::type>
class NumericInplaceIndexSortKernel : public InplaceIndexSortKernel {
 public:
  using T = typename TYPE::c_type;

  explicit NumericInplaceIndexSortKernel(arrow::MemoryPool *pool) : InplaceIndexSortKernel(pool) {}

  arrow::Status Sort(std::shared_ptr<arrow::Array> &values, std::shared_ptr<arrow::UInt64Array> &offsets) override {
    auto array = std::static_pointer_cast<arrow::NumericArray<TYPE>>(values);
    std::shared_ptr<arrow::ArrayData> data = array->data();
    // get the first buffer as a mutable buffer
    T *left_data = data->GetMutableValues<T>(1);
    int64_t length = values->length();
    int64_t buf_size = length * sizeof(uint64_t);

    arrow::Result<std::unique_ptr<arrow::Buffer>> result = AllocateBuffer(buf_size + 1, pool_);
    const arrow::Status &status = result.status();
    if (!status.ok()) {
      LOG(FATAL) << "Failed to allocate sort indices - " << status.message();
      return status;
    }
    std::shared_ptr<arrow::Buffer> indices_buf = std::move(result.ValueOrDie());

    auto *indices_begin = reinterpret_cast<int64_t *>(indices_buf->mutable_data());
    for (int64_t i = 0; i < length; i++) {
      indices_begin[i] = i;
    }
    cylon::util::quicksort(left_data, 0, length, indices_begin);
    offsets = std::make_shared<arrow::UInt64Array>(length, indices_buf);
    return arrow::Status::OK();
  }
};

using UInt8ArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::UInt8Type>;
using UInt16ArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::UInt16Type>;
using UInt32ArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::UInt32Type>;
using UInt64ArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::UInt64Type>;
using Int8ArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::Int8Type>;
using Int16ArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::Int16Type>;
using Int32ArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::Int32Type>;
using Int64ArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::Int64Type>;
using HalfFloatArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::HalfFloatType>;
using FloatArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::FloatType>;
using DoubleArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::DoubleType>;

arrow::Status SortIndicesInPlace(arrow::MemoryPool *memory_pool,
                                 std::shared_ptr<arrow::Array> &values,
                                 std::shared_ptr<arrow::UInt64Array> &offsets);

}

#endif //CYLON_ARROW_KERNELS_H

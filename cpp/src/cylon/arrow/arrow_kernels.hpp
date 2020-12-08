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
  virtual int Split(std::shared_ptr<arrow::Array> &values,
                    const std::vector<int64_t> &partitions,
                    const std::vector<int32_t> &targets,
                    std::unordered_map<int, std::shared_ptr<arrow::Array>> &out,
                    std::vector<uint32_t> &counts) = 0;

  virtual Status Split(const std::shared_ptr<arrow::ChunkedArray> &values,
                       const std::vector<uint32_t> &target_partitions,
                       uint32_t num_partitions,
                       const std::vector<uint32_t> &counts,
                       std::vector<std::shared_ptr<arrow::Array>> &output) = 0;

 protected:
  arrow::MemoryPool *pool_;
};

template<typename TYPE>
class ArrowArrayNumericSplitKernel : public ArrowArraySplitKernel {
 public:
  explicit ArrowArrayNumericSplitKernel(arrow::MemoryPool *pool) : ArrowArraySplitKernel(pool) {}

  using ARROW_ARRAY_T = typename arrow::TypeTraits<TYPE>::ArrayType;
  using ARROW_BUILDER_T = typename arrow::TypeTraits<TYPE>::BuilderType;

  int Split(std::shared_ptr<arrow::Array> &values,
            const std::vector<int64_t> &partitions,
            const std::vector<int32_t> &targets,
            std::unordered_map<int, std::shared_ptr<arrow::Array>> &out,
            std::vector<uint32_t> &counts) override {
    auto reader = std::static_pointer_cast<ARROW_ARRAY_T>(values);
    std::vector<std::shared_ptr<ARROW_BUILDER_T>> builders;

    for (size_t i = 0; i < targets.size(); i++) {
      std::shared_ptr<ARROW_BUILDER_T> b = std::make_shared<ARROW_BUILDER_T>(pool_);
      b->Reserve(counts[i]);
      builders.push_back(b);
    }

    size_t kI = partitions.size();
    for (size_t i = 0; i < kI; i++) {
      const std::shared_ptr<ARROW_BUILDER_T> &b = builders[partitions[i]];
      b->UnsafeAppend(reader->Value(i));
    }

    for (long target : targets) {
      const std::shared_ptr<ARROW_BUILDER_T> &b = builders[target];
      std::shared_ptr<arrow::Array> array;
      b->Finish(&array);
      out.insert(std::pair<int, std::shared_ptr<arrow::Array>>(target, array));
    }
    return 0;
  }

  Status Split(const std::shared_ptr<arrow::ChunkedArray> &values,
               const std::vector<uint32_t> &target_partitions,
               uint32_t num_partitions,
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
      for (int64_t i = 0; i < arr_len; i++) {
        builders[target_partitions[offset + i]]->UnsafeAppend(casted_array->Value(i));
      }
      offset += arr_len;
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

// todo: this can be replaced by numeric kernel ( )
class FixedBinaryArraySplitKernel : public ArrowArraySplitKernel {
 public:
  explicit FixedBinaryArraySplitKernel(arrow::MemoryPool *pool) : ArrowArraySplitKernel(pool) {}

  int Split(std::shared_ptr<arrow::Array> &values,
            const std::vector<int64_t> &partitions,
            const std::vector<int32_t> &targets,
            std::unordered_map<int, std::shared_ptr<arrow::Array>> &out,
            std::vector<uint32_t> &counts) override;

  Status Split(const std::shared_ptr<arrow::ChunkedArray> &values,
               const std::vector<uint32_t> &target_partitions,
               uint32_t num_partitions,
               const std::vector<uint32_t> &counts,
               std::vector<std::shared_ptr<arrow::Array>> &output) override;
};

template<typename TYPE>
class BinaryArraySplitKernel : public ArrowArraySplitKernel {
 public:
  using ARROW_ARRAY_T = typename arrow::TypeTraits<TYPE>::ArrayType;
  using ARROW_BUILDER_T = typename arrow::TypeTraits<TYPE>::BuilderType;

  explicit BinaryArraySplitKernel(arrow::MemoryPool *pool) : ArrowArraySplitKernel(pool) {
//    std::function<>
  }

  Status Split(const std::shared_ptr<arrow::ChunkedArray> &values,
               const std::vector<uint32_t> &target_partitions,
               uint32_t num_partitions,
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

    return Status::OK();
  }

  int Split(std::shared_ptr<arrow::Array> &values,
            const std::vector<int64_t> &partitions,
            const std::vector<int32_t> &targets,
            std::unordered_map<int, std::shared_ptr<arrow::Array>> &out,
            std::vector<uint32_t> &counts) override {
    auto reader = std::static_pointer_cast<ARROW_ARRAY_T>(values);
    std::unordered_map<int, std::shared_ptr<ARROW_BUILDER_T>> builders;

    for (int it : targets) {
      std::shared_ptr<ARROW_BUILDER_T> b = std::make_shared<ARROW_BUILDER_T>(pool_);
      builders.insert(std::pair<int, std::shared_ptr<ARROW_BUILDER_T>>(it, b));
    }

    for (size_t i = 0; i < partitions.size(); i++) {
      const std::shared_ptr<ARROW_BUILDER_T> &b = builders[partitions.at(i)];
      int length = 0;
      const uint8_t *value = reader->GetValue(i, &length);
      if (b->Append(value, length) != arrow::Status::OK()) {
        LOG(FATAL) << "Failed to merge";
        return -1;
      }
    }

    for (int it : targets) {
      const std::shared_ptr<ARROW_BUILDER_T> &b = builders[it];
      std::shared_ptr<arrow::Array> array;
      if (b->Finish(&array) != arrow::Status::OK()) {
        LOG(FATAL) << "Failed to merge";
        return -1;
      }
      out.insert(std::pair<int, std::shared_ptr<arrow::Array>>(it, array));
    }
    return 0;
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

cylon::Status CreateSplitter(const std::shared_ptr<arrow::DataType> &type,
                             arrow::MemoryPool *pool,
                             std::shared_ptr<ArrowArraySplitKernel> *out);

class ArrowArraySortKernel {
 public:
  explicit ArrowArraySortKernel(arrow::MemoryPool *pool) : pool_(pool) {}

  /**
   * Sort the values in the column and return an array with the indices
   * @param ctx
   * @param values
   * @param targets
   * @param out_length
   * @param out
   * @return
   */
  virtual int Sort(std::shared_ptr<arrow::Array> values, std::shared_ptr<arrow::Array> *out) = 0;

 protected:
  arrow::MemoryPool *pool_;
};

template<typename TYPE>
class ArrowArrayNumericSortKernel : public ArrowArraySortKernel {
 public:
  using T = typename TYPE::c_type;

  explicit ArrowArrayNumericSortKernel(arrow::MemoryPool *pool) : ArrowArraySortKernel(pool) {}

  int Sort(std::shared_ptr<arrow::Array> values,
		   std::shared_ptr<arrow::Array> *offsets) override {
	auto array = std::static_pointer_cast<arrow::NumericArray<TYPE>>(values);
	const T *left_data = array->raw_values();
	int64_t buf_size = values->length() * sizeof(uint64_t);

    arrow::Result<std::unique_ptr<arrow::Buffer>> result = AllocateBuffer(buf_size + 1, pool_);
	const arrow::Status &status = result.status();
	if (!status.ok()) {
	  LOG(FATAL) << "Failed to allocate sort indices - " << status.message();
	  return -1;
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
	*offsets = std::make_shared<arrow::UInt64Array>(values->length(), indices_buf);
	return 0;
  }
};

using UInt8ArraySorter = ArrowArrayNumericSortKernel<arrow::UInt8Type>;
using UInt16ArraySorter = ArrowArrayNumericSortKernel<arrow::UInt16Type>;
using UInt32ArraySorter = ArrowArrayNumericSortKernel<arrow::UInt32Type>;
using UInt64ArraySorter = ArrowArrayNumericSortKernel<arrow::UInt64Type>;
using Int8ArraySorter = ArrowArrayNumericSortKernel<arrow::Int8Type>;
using Int16ArraySorter = ArrowArrayNumericSortKernel<arrow::Int16Type>;
using Int32ArraySorter = ArrowArrayNumericSortKernel<arrow::Int32Type>;
using Int64ArraySorter = ArrowArrayNumericSortKernel<arrow::Int64Type>;
using HalfFloatArraySorter = ArrowArrayNumericSortKernel<arrow::HalfFloatType>;
using FloatArraySorter = ArrowArrayNumericSortKernel<arrow::FloatType>;
using DoubleArraySorter = ArrowArrayNumericSortKernel<arrow::DoubleType>;

arrow::Status SortIndices(arrow::MemoryPool *memory_pool, std::shared_ptr<arrow::Array> &values,
                          std::shared_ptr<arrow::Array> *offsets);

class ArrowArrayInplaceSortKernel {
 public:
  explicit ArrowArrayInplaceSortKernel(arrow::MemoryPool *pool) : pool_(pool) {}

  /**
   * Sort the values in the column and return an array with the indices
   * @param ctx
   * @param values
   * @param targets
   * @param out_length
   * @param out
   * @return
   */
  virtual int Sort(std::shared_ptr<arrow::Array> values, std::shared_ptr<arrow::UInt64Array> *out) = 0;
 protected:
  arrow::MemoryPool *pool_;
};

template<typename TYPE>
class ArrowArrayInplaceNumericSortKernel : public ArrowArrayInplaceSortKernel {
 public:
  using T = typename TYPE::c_type;

  explicit ArrowArrayInplaceNumericSortKernel(arrow::MemoryPool *pool) :
      ArrowArrayInplaceSortKernel(pool) {}

  int Sort(std::shared_ptr<arrow::Array> values,
           std::shared_ptr<arrow::UInt64Array> *offsets) override {
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
      return -1;
    }
    std::shared_ptr<arrow::Buffer> indices_buf = std::move(result.ValueOrDie());

    auto *indices_begin = reinterpret_cast<int64_t *>(indices_buf->mutable_data());
    for (int64_t i = 0; i < length; i++) {
      indices_begin[i] = i;
    }
    cylon::util::quicksort(left_data, 0, length, indices_begin);
    *offsets = std::make_shared<arrow::UInt64Array>(length, indices_buf);
    return 0;
  }
};

using UInt8ArrayInplaceSorter = ArrowArrayInplaceNumericSortKernel<arrow::UInt8Type>;
using UInt16ArrayInplaceSorter = ArrowArrayInplaceNumericSortKernel<arrow::UInt16Type>;
using UInt32ArrayInplaceSorter = ArrowArrayInplaceNumericSortKernel<arrow::UInt32Type>;
using UInt64ArrayInplaceSorter = ArrowArrayInplaceNumericSortKernel<arrow::UInt64Type>;
using Int8ArrayInplaceSorter = ArrowArrayInplaceNumericSortKernel<arrow::Int8Type>;
using Int16ArrayInplaceSorter = ArrowArrayInplaceNumericSortKernel<arrow::Int16Type>;
using Int32ArrayInplaceSorter = ArrowArrayInplaceNumericSortKernel<arrow::Int32Type>;
using Int64ArrayInplaceSorter = ArrowArrayInplaceNumericSortKernel<arrow::Int64Type>;
using HalfFloatArrayInplaceSorter = ArrowArrayInplaceNumericSortKernel<arrow::HalfFloatType>;
using FloatArrayInplaceSorter = ArrowArrayInplaceNumericSortKernel<arrow::FloatType>;
using DoubleArrayInplaceSorter = ArrowArrayInplaceNumericSortKernel<arrow::DoubleType>;

arrow::Status SortIndicesInPlace(arrow::MemoryPool *memory_pool,
                                 std::shared_ptr<arrow::Array> &values,
                                 std::shared_ptr<arrow::UInt64Array> *offsets);

}

#endif //CYLON_ARROW_KERNELS_H

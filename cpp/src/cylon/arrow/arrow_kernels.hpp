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
#include "../status.hpp"
#include "util/sort.hpp"

namespace cylon {

class ArrowArraySplitKernel {
 public:
  explicit ArrowArraySplitKernel(std::shared_ptr<arrow::DataType> type,
								 arrow::MemoryPool *pool) : type_(type), pool_(pool) {}

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
 protected:
  std::shared_ptr<arrow::DataType> type_;
  arrow::MemoryPool *pool_;
};

template<typename TYPE>
class ArrowArrayNumericSplitKernel : public ArrowArraySplitKernel {
 public:
  explicit ArrowArrayNumericSplitKernel(std::shared_ptr<arrow::DataType> type,
										arrow::MemoryPool *pool) :
	  ArrowArraySplitKernel(type, pool) {}

  int Split(std::shared_ptr<arrow::Array> &values,
            const std::vector<int64_t> &partitions,
            const std::vector<int32_t> &targets,
            std::unordered_map<int, std::shared_ptr<arrow::Array>> &out,
            std::vector<uint32_t> &counts) override {
	auto reader =
		std::static_pointer_cast<arrow::NumericArray<TYPE>>(values);
	std::vector<std::shared_ptr<arrow::NumericBuilder<TYPE>>> builders;

    for (size_t i = 0; i < targets.size(); i++) {
	  std::shared_ptr<arrow::NumericBuilder<TYPE>> b =
	      std::make_shared<arrow::NumericBuilder<TYPE>>(type_, pool_);
      b->Reserve(counts[i]);
	  builders.push_back(b);
	}

    size_t kI = partitions.size();
    for (size_t i = 0; i < kI; i++) {
	  std::shared_ptr<arrow::NumericBuilder<TYPE>> b = builders[partitions[i]];
	  b->UnsafeAppend(reader->Value(i));
	}

	for (long target : targets) {
	  std::shared_ptr<arrow::NumericBuilder<TYPE>> b = builders[target];
	  std::shared_ptr<arrow::Array> array;
	  b->Finish(&array);
	  out.insert(std::pair<int, std::shared_ptr<arrow::Array>>(target, array));
	}
	return 0;
  }
};

class FixedBinaryArraySplitKernel : public ArrowArraySplitKernel {
 public:
  explicit FixedBinaryArraySplitKernel(std::shared_ptr<arrow::DataType> type,
									   arrow::MemoryPool *pool) :
	  ArrowArraySplitKernel(type, pool) {}

  int Split(std::shared_ptr<arrow::Array> &values,
            const std::vector<int64_t> &partitions,
            const std::vector<int32_t> &targets,
            std::unordered_map<int, std::shared_ptr<arrow::Array>> &out,
            std::vector<uint32_t> &counts) override;
};

class BinaryArraySplitKernel : public ArrowArraySplitKernel {
 public:
  explicit BinaryArraySplitKernel(std::shared_ptr<arrow::DataType> type,
								  arrow::MemoryPool *pool) :
	  ArrowArraySplitKernel(type, pool) {}

  int Split(std::shared_ptr<arrow::Array> &values,
            const std::vector<int64_t> &partitions,
            const std::vector<int32_t> &targets,
            std::unordered_map<int, std::shared_ptr<arrow::Array>> &out,
            std::vector<uint32_t> &counts) override;
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

class ArrowArrayStreamingSplitKernel {
 public:
  explicit ArrowArrayStreamingSplitKernel(const std::shared_ptr<arrow::DataType> &type,
                                 const std::vector<int32_t> &targets,
                                 arrow::MemoryPool *pool) : type_(type), pool_(pool),
                                 targets_(targets) {}

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
                    const std::vector<uint32_t> &cnts) = 0;

  /**
   * Finish the split
   * @param out
   * @param counts
   * @return
   */
  virtual int finish(std::unordered_map<int, std::shared_ptr<arrow::Array>> &out) = 0;
 protected:
  std::shared_ptr<arrow::DataType> type_;
  arrow::MemoryPool *pool_;
  const std::vector<int32_t> targets_;
};

template<typename TYPE>
class ArrowArrayStreamingNumericSplitKernel : public ArrowArrayStreamingSplitKernel {
 private:
  std::vector<std::shared_ptr<arrow::NumericBuilder<TYPE>>> builders_;
 public:
  explicit ArrowArrayStreamingNumericSplitKernel(const std::shared_ptr<arrow::DataType> &type,
                                                 const std::vector<int32_t> &targets,
                                        arrow::MemoryPool *pool) :
      ArrowArrayStreamingSplitKernel(type, targets, pool) {
    for (size_t i = 0; i < targets.size(); i++) {
      std::shared_ptr<arrow::NumericBuilder<TYPE>> b =
          std::make_shared<arrow::NumericBuilder<TYPE>>(type_, pool_);
      builders_.push_back(b);
    }
  }

  int Split(std::shared_ptr<arrow::Array> &values,
            const std::vector<int64_t> &partitions,
            const std::vector<uint32_t> &cnts) override {
    auto reader =
        std::static_pointer_cast<arrow::NumericArray<TYPE>>(values);
    size_t kI = partitions.size();
    size_t kSize = builders_.size();
    for (size_t i = 0; i < kSize; i++) {
      builders_[i]->Reserve(cnts[i]);
    }

    for (size_t i = 0; i < kI; i++) {
      std::shared_ptr<arrow::NumericBuilder<TYPE>> b = builders_[partitions[i]];
      b->Append(reader->Value(i));
    }
    return 0;
  }

  int finish(std::unordered_map<int, std::shared_ptr<arrow::Array>> &out) override {
    for (long target : targets_) {
      std::shared_ptr<arrow::NumericBuilder<TYPE>> b = builders_[target];
      std::shared_ptr<arrow::Array> array;
      b->Finish(&array);
      out.insert(std::pair<int, std::shared_ptr<arrow::Array>>(target, array));
    }
    return 0;
  }
};

class FixedBinaryArrayStreamingSplitKernel : public ArrowArrayStreamingSplitKernel {
 private:
  std::unordered_map<int, std::shared_ptr<arrow::FixedSizeBinaryBuilder>> builders_;
 public:
  explicit FixedBinaryArrayStreamingSplitKernel(const std::shared_ptr<arrow::DataType>& type,
                                                const std::vector<int32_t> &targets,
                                       arrow::MemoryPool *pool);

  int Split(std::shared_ptr<arrow::Array> &values,
            const std::vector<int64_t> &partitions,
            const std::vector<uint32_t> &cnts) override;

  int finish(std::unordered_map<int, std::shared_ptr<arrow::Array>> &out) override;
};

class BinaryArrayStreamingSplitKernel : public ArrowArrayStreamingSplitKernel {
 private:
  std::unordered_map<int, std::shared_ptr<arrow::BinaryBuilder>> builders_;
 public:
  explicit BinaryArrayStreamingSplitKernel(const std::shared_ptr<arrow::DataType>& type,
                                           const std::vector<int32_t> &targets,
                                  arrow::MemoryPool *pool);

  int Split(std::shared_ptr<arrow::Array> &values,
            const std::vector<int64_t> &partitions,
            const std::vector<uint32_t> &cnts) override;

  int finish(std::unordered_map<int, std::shared_ptr<arrow::Array>> &out) override;
};

using UInt8ArrayStreamingSplitter = ArrowArrayStreamingNumericSplitKernel<arrow::UInt8Type>;
using UInt16ArrayStreamingSplitter = ArrowArrayStreamingNumericSplitKernel<arrow::UInt16Type>;
using UInt32ArrayStreamingSplitter = ArrowArrayStreamingNumericSplitKernel<arrow::UInt32Type>;
using UInt64ArrayStreamingSplitter = ArrowArrayStreamingNumericSplitKernel<arrow::UInt64Type>;

using Int8ArrayStreamingSplitter = ArrowArrayStreamingNumericSplitKernel<arrow::Int8Type>;
using Int16ArrayStreamingSplitter = ArrowArrayStreamingNumericSplitKernel<arrow::Int16Type>;
using Int32ArrayStreamingSplitter = ArrowArrayStreamingNumericSplitKernel<arrow::Int32Type>;
using Int64ArrayStreamingSplitter = ArrowArrayStreamingNumericSplitKernel<arrow::Int64Type>;

using HalfFloatArrayStreamingSplitter = ArrowArrayStreamingNumericSplitKernel<arrow::HalfFloatType>;
using FloatArrayStreamingSplitter = ArrowArrayStreamingNumericSplitKernel<arrow::FloatType>;
using DoubleArrayStreamingSplitter = ArrowArrayStreamingNumericSplitKernel<arrow::DoubleType>;

cylon::Status CreateStreamingSplitter(const std::shared_ptr<arrow::DataType> &type,
                                      const std::vector<int32_t> &targets,
                             arrow::MemoryPool *pool,
                             std::shared_ptr<ArrowArrayStreamingSplitKernel> *out);

class ArrowArraySortKernel {
 public:
  explicit ArrowArraySortKernel(std::shared_ptr<arrow::DataType> type,
								arrow::MemoryPool *pool) : type_(type), pool_(pool) {}

  /**
   * Sort the values in the column and return an array with the indices
   * @param ctx
   * @param values
   * @param targets
   * @param out_length
   * @param out
   * @return
   */
  virtual int Sort(std::shared_ptr<arrow::Array> values,
				   std::shared_ptr<arrow::Array> *out) = 0;
 protected:
  std::shared_ptr<arrow::DataType> type_;
  arrow::MemoryPool *pool_;
};

template<typename TYPE>
class ArrowArrayNumericSortKernel : public ArrowArraySortKernel {
 public:
  using T = typename TYPE::c_type;

  explicit ArrowArrayNumericSortKernel(std::shared_ptr<arrow::DataType> type,
									   arrow::MemoryPool *pool) :
	  ArrowArraySortKernel(type, pool) {}

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

arrow::Status SortIndices(arrow::MemoryPool *memory_pool, std::shared_ptr<arrow::Array> values,
						  std::shared_ptr<arrow::Array> *offsets);

class ArrowArrayInplaceSortKernel {
 public:
  explicit ArrowArrayInplaceSortKernel(std::shared_ptr<arrow::DataType> type,
                                       arrow::MemoryPool *pool) : type_(type), pool_(pool) {}

  /**
   * Sort the values in the column and return an array with the indices
   * @param ctx
   * @param values
   * @param targets
   * @param out_length
   * @param out
   * @return
   */
  virtual int Sort(std::shared_ptr<arrow::Array> values,
                   std::shared_ptr<arrow::UInt64Array> *out) = 0;
 protected:
  std::shared_ptr<arrow::DataType> type_;
  arrow::MemoryPool *pool_;
};

template<typename TYPE>
class ArrowArrayInplaceNumericSortKernel : public ArrowArrayInplaceSortKernel {
 public:
  using T = typename TYPE::c_type;

  explicit ArrowArrayInplaceNumericSortKernel(std::shared_ptr<arrow::DataType> type,
                                              arrow::MemoryPool *pool) :
      ArrowArrayInplaceSortKernel(type, pool) {}

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
                                 std::shared_ptr<arrow::Array> values,
                                 std::shared_ptr<arrow::UInt64Array> *offsets);

}

#endif //CYLON_ARROW_KERNELS_H

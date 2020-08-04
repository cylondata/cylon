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
					std::unordered_map<int, std::shared_ptr<arrow::Array>> &out) = 0;
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
			std::unordered_map<int, std::shared_ptr<arrow::Array>> &out) override {
	auto reader =
		std::static_pointer_cast<arrow::NumericArray<TYPE>>(values);
	std::unordered_map<int, std::shared_ptr<arrow::NumericBuilder<TYPE>>> builders;

	for (long target : targets) {
	  std::shared_ptr<arrow::NumericBuilder<TYPE>> b = std::make_shared<arrow::NumericBuilder<TYPE>>(type_, pool_);
	  builders.insert(std::pair<int, std::shared_ptr<arrow::NumericBuilder<TYPE>>>(target, b));
	}

	for (size_t i = 0; i < partitions.size(); i++) {
	  std::shared_ptr<arrow::NumericBuilder<TYPE>> b = builders[partitions.at(i)];
	  b->Append(reader->Value(i));
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
			std::unordered_map<int, std::shared_ptr<arrow::Array>> &out) override;
};

class BinaryArraySplitKernel : public ArrowArraySplitKernel {
 public:
  explicit BinaryArraySplitKernel(std::shared_ptr<arrow::DataType> type,
								  arrow::MemoryPool *pool) :
	  ArrowArraySplitKernel(type, pool) {}

  int Split(std::shared_ptr<arrow::Array> &values,
			const std::vector<int64_t> &partitions,
			const std::vector<int32_t> &targets,
			std::unordered_map<int, std::shared_ptr<arrow::Array>> &out) override;
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
	std::shared_ptr<arrow::Buffer> indices_buf;
	int64_t buf_size = values->length() * sizeof(uint64_t);
	arrow::Status status = AllocateBuffer(arrow::default_memory_pool(), buf_size + 1, &indices_buf);
	if (status != arrow::Status::OK()) {
	  LOG(FATAL) << "Failed to allocate sort indices - " << status.message();
	  return -1;
	}
	auto *indices_begin = reinterpret_cast<int64_t *>(indices_buf->mutable_data());
	for (int64_t i = 0; i < values->length(); i++) {
	  indices_begin[i] = i;
	}
	int64_t *indices_end = indices_begin + values->length();
	// LOG(INFO) << "Length " << values->length() << " ind " << indices_begin << "nd " << indices_begin + values->length();
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

}

#endif //CYLON_ARROW_KERNELS_H

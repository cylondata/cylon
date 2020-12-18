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

#include "../status.hpp"

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

std::unique_ptr<ArrowArraySplitKernel> CreateSplitter(const std::shared_ptr<arrow::DataType> &type,
                                                      arrow::MemoryPool *pool);

// -----------------------------------------------------------------------------

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

/**
 * sort indices
 * @param memory_pool
 * @param values
 * @param offsets
 * @return
 */
arrow::Status SortIndices(arrow::MemoryPool *memory_pool, std::shared_ptr<arrow::Array> &values,
                          std::shared_ptr<arrow::Array> &offsets);

// -----------------------------------------------------------------------------

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

/**
 * sort indices in-place
 * @param memory_pool
 * @param values
 * @param offsets
 * @return
 */
arrow::Status SortIndicesInPlace(arrow::MemoryPool *memory_pool,
                                 std::shared_ptr<arrow::Array> &values,
                                 std::shared_ptr<arrow::UInt64Array> &offsets);

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
    auto reader = std::static_pointer_cast<arrow::NumericArray<TYPE>>(values);
    size_t kI = partitions.size();
//    size_t kSize = builders_.size();
//    for (size_t i = 0; i < kSize; i++) {
//      builders_[i]->Reserve(cnts[i]);
//    }

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
  explicit FixedBinaryArrayStreamingSplitKernel(const std::shared_ptr<arrow::DataType> &type,
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
  explicit BinaryArrayStreamingSplitKernel(const std::shared_ptr<arrow::DataType> &type,
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

Status CreateStreamingSplitter(const std::shared_ptr<arrow::DataType> &type,
                               const std::vector<int32_t> &targets,
                               arrow::MemoryPool *pool,
                               std::shared_ptr<ArrowArrayStreamingSplitKernel> *out);

}

#endif //CYLON_ARROW_KERNELS_H

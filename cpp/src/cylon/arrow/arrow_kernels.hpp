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

// -----------------------------------------------------------------------------

class StreamingSplitKernel {
 public:
  /**
   * Merge the values in the column and return an array
   * @param ctx
   * @param values
   * @param targets
   * @param out_length
   * @param out
   * @return
   */
  virtual Status Split(const std::shared_ptr<arrow::Array> &values,
                       const std::vector<uint32_t> &partitions,
                       const std::vector<uint32_t> &counts) = 0; // todo make count int64_t

  /**
   * Finish the split
   * @param out
   * @param counts
   * @return
   */
  virtual Status Finish(std::vector<std::shared_ptr<arrow::Array>> &out) = 0;
};

std::unique_ptr<StreamingSplitKernel> CreateStreamingSplitter(const std::shared_ptr<arrow::DataType> &type,
                                                              int32_t targets,
                                                              arrow::MemoryPool *pool);

}

#endif //CYLON_ARROW_KERNELS_H

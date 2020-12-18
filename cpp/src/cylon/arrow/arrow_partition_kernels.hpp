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

#ifndef CYLON_ARROW_PARTITION_KERNELS_H
#define CYLON_ARROW_PARTITION_KERNELS_H

#include <arrow/api.h>
#include <glog/logging.h>

#include "../ctx/cylon_context.hpp"

namespace cylon {

class PartitionKernel {
 public:
  virtual ~PartitionKernel() = default;

  /**
   * partitions idx_col to provided number of partitions
   * @param idx_col
   * @param num_partitions
   * @param target_partitions
   * @param partition_histogram
   * @return
   */
  Status Partition(const std::shared_ptr<arrow::Array> &idx_col,
                   uint32_t num_partitions,
                   std::vector<uint32_t> &target_partitions,
                   std::vector<uint32_t> &partition_histogram) {
    return Partition(std::make_shared<arrow::ChunkedArray>(idx_col),
                     num_partitions,
                     target_partitions,
                     partition_histogram);
  }

  virtual Status Partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                           uint32_t num_partitions,
                           std::vector<uint32_t> &target_partitions,
                           std::vector<uint32_t> &partition_histogram) = 0;
};

/**
 * Partition kernel based on hash value
 */
class HashPartitionKernel : public PartitionKernel {
 public:

  /**
   * Bulk up date of partial_hashes vector based on the idx col (can be used for building a composite hash for multiple
   * columns)
   * @param idx_col
   * @param partial_hashes
   * @return
   */
  Status UpdateHash(const std::shared_ptr<arrow::Array> &idx_col, std::vector<uint32_t> &partial_hashes) {
    return UpdateHash(std::make_shared<arrow::ChunkedArray>(idx_col), partial_hashes);
  }

  virtual Status UpdateHash(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                            std::vector<uint32_t> &partial_hashes) = 0;

  /**
   * hash value of a particular index of a column (using this while iterating a column would be sub-optimal. consider
   * using cylon::UpdateHash method)
   * @param values
   * @param index
   * @return
   */
  virtual uint32_t ToHash(const std::shared_ptr<arrow::Array> &values, int64_t index) = 0;
};


/**
 * creates hash partition kernel
 * @param data_type
 * @return
 */
std::unique_ptr<HashPartitionKernel> CreateHashPartitionKernel(const std::shared_ptr<arrow::DataType> &data_type);


class RowHashingKernel {
 private:
  std::vector<std::shared_ptr<HashPartitionKernel>> hash_kernels;
 public:
  explicit RowHashingKernel(const std::vector<std::shared_ptr<arrow::Field>> &fields);

  int32_t Hash(const std::shared_ptr<arrow::Table> &table, int64_t row) const;
};

/**
 * create range partition
 * @param data_type
 * @param num_partitions
 * @param ctx
 * @param ascending
 * @param num_samples
 * @param num_bins
 * @return
 */
std::unique_ptr<PartitionKernel> CreateRangePartitionKernel(const std::shared_ptr<arrow::DataType> &data_type,
                                                            std::shared_ptr<CylonContext> &ctx,
                                                            bool ascending,
                                                            uint64_t num_samples,
                                                            uint32_t num_bins);
}  // namespace cylon

#endif //CYLON_ARROW_PARTITION_KERNELS_H

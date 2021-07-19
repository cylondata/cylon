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

#ifndef CYLON_SRC_CYLON_OPS_KERNELS_PARTITION_HPP_
#define CYLON_SRC_CYLON_OPS_KERNELS_PARTITION_HPP_

#include <cylon/arrow/arrow_partition_kernels.hpp>
#include <cylon/arrow/arrow_kernels.hpp>
#include <cylon/table.hpp>

namespace cylon {
namespace kernel {
class StreamingHashPartitionKernel {
 public:
  StreamingHashPartitionKernel(const std::shared_ptr<CylonContext> &ctx,
                               const std::shared_ptr<arrow::Schema> &schema,
                               int num_partitions,
                               const std::vector<int> &hash_columns);

  Status Process(int tag, const std::shared_ptr<Table> &table);

  Status Finish(std::vector<std::shared_ptr<Table>> &partitioned_tables);

 private:
  int num_partitions;
  std::vector<int> hash_columns;
  std::shared_ptr<arrow::Schema> schema;
  std::shared_ptr<CylonContext> ctx;

  std::vector<std::unique_ptr<HashPartitionKernel>> partition_kernels = {};
  std::vector<std::unique_ptr<StreamingSplitKernel>> split_kernels = {};
  std::vector<uint32_t> temp_partitions = {}, temp_partition_hist = {};
};
}
}

#endif //CYLON_SRC_CYLON_OPS_KERNELS_PARTITION_HPP_

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

#ifndef CYLON_SRC_CYLON_OPS_PARTITION_OP_HPP_
#define CYLON_SRC_CYLON_OPS_PARTITION_OP_HPP_
#include <chrono>

#include <arrow/arrow_partition_kernels.hpp>

#include "kernels/partition.hpp"
#include "ops/api/parallel_op.hpp"

namespace cylon {

class PartitionOpConfig {
 public:
  PartitionOpConfig(int num_partitions, std::vector<int> hash_columns)
      : num_partitions(num_partitions), hash_columns(std::move(hash_columns)) {}

  int num_partitions;
  std::vector<int> hash_columns;
};

class PartitionOp : public Op {
 private:
  std::chrono::high_resolution_clock::time_point start;
  bool started_time = false;

  PartitionOpConfig config;
 public:
  PartitionOp(const std::shared_ptr<cylon::CylonContext> &ctx,
              const std::shared_ptr<arrow::Schema> &schema,
              int id,
              const ResultsCallback &callback,
              const PartitionOpConfig &config);
  bool Execute(int tag, std::shared_ptr<Table> &table) override;
  void OnParentsFinalized() override;
  bool Finalize() override;
};
}

#endif //CYLON_SRC_CYLON_OPS_PARTITION_OP_HPP_

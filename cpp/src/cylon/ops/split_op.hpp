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

#ifndef CYLON_SRC_CYLON_OPS_SPLIT_OP_H_
#define CYLON_SRC_CYLON_OPS_SPLIT_OP_H_

#include <vector>
#include <map>

#include "../arrow/arrow_kernels.hpp"

#include "parallel_op.hpp"
#include "partition_op.hpp"

namespace cylon {

struct SplitOpConfig {
 public:
  SplitOpConfig(int num_partitions, std::vector<int> hash_columns)
      : num_partitions(num_partitions), hash_columns(std::move(hash_columns)) {}

  static std::shared_ptr<SplitOpConfig> Make(int no_partitions, const std::vector<int> &hash_cols){
    return std::make_shared<SplitOpConfig>(no_partitions, hash_cols);
  }

  int num_partitions;
  std::vector<int> hash_columns;
};

// todo this op is so similar to partition. may be call repartition/ streaming repartition
class SplitOp : public Op {
 private:
  kernel::StreamingHashPartitionKernel partition_kernel;
  std::chrono::high_resolution_clock::time_point start;
  bool started_time = false;
  long exec_time = 0;
  long count = 0;
 public:
  SplitOp(const std::shared_ptr<CylonContext> &ctx,
          const std::shared_ptr<arrow::Schema> &schema,
          int32_t id,
          const ResultsCallback &callback,
          const SplitOpConfig &cfg);

  bool Execute(int tag, std::shared_ptr<Table> &table) override;

  void OnParentsFinalized() override;

  bool Finalize() override;
};
}  // namespace cylon

#endif //CYLON_SRC_CYLON_OPS_SPLIT_OP_H_

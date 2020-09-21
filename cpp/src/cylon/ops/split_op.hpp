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

#include "parallel_op.hpp"
#include "partition_op.hpp"

namespace cylon {

class SplitOpConfig {
 private:
  int no_of_partitions;
  std::shared_ptr<std::vector<int>> hash_columns;

 public:
  SplitOpConfig(int no_of_partitions, std::shared_ptr<std::vector<int>> hash_columns);
  int NoOfPartitions();
  std::shared_ptr<std::vector<int>> HashColumns();

  static std::shared_ptr<SplitOpConfig> Make(int no_partitions, const std::vector<int> &hash_cols);
};

class SplitOp : public Op {
 private:
  std::vector<std::shared_ptr<ArrowArrayStreamingSplitKernel>> received_tables_;
  std::shared_ptr<SplitOpConfig> config_;
  int hash_column_;
  std::vector<int> targets;
  std::vector<int> hash_targets;
  std::chrono::high_resolution_clock::time_point start;
  bool started_time = false;
  long exec_time = 0;
 public:
  SplitOp(const std::shared_ptr<CylonContext> &ctx,
          const std::shared_ptr<arrow::Schema> &schema,
          int32_t id,
          const std::shared_ptr<ResultsCallback> &callback,
          const std::shared_ptr<SplitOpConfig> &cfg);

  bool Execute(int tag, std::shared_ptr<Table> table) override;

  void OnParentsFinalized() override;

  bool Finalize() override;
};
}  // namespace cylon

#endif //CYLON_SRC_CYLON_OPS_SPLIT_OP_H_

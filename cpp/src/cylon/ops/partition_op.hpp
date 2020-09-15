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
#include "parallel_op.hpp"

namespace cylon {

class PartitionOpConfig {
 private:
  int no_of_partitions;
  std::shared_ptr<std::vector<int>> hash_columns;

 public:
  PartitionOpConfig(int no_of_partitions, std::shared_ptr<std::vector<int>> hash_columns);
  int NoOfPartitions();
  std::shared_ptr<std::vector<int>> HashColumns();
};

class PartitionOp : public Op {
 private:
  std::shared_ptr<PartitionOpConfig> config;

 public:
  PartitionOp(const std::shared_ptr<cylon::CylonContext> &ctx,
              const std::shared_ptr<arrow::Schema> &schema,
              int id,
              const std::shared_ptr<ResultsCallback> &callback,
              const std::shared_ptr<PartitionOpConfig> &config);
  bool Execute(int tag, std::shared_ptr<Table> table) override;
  void OnParentsFinalized() override;
  bool Finalize() override;
};
}

#endif //CYLON_SRC_CYLON_OPS_PARTITION_OP_HPP_

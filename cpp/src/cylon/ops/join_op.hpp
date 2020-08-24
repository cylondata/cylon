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

#ifndef CYLON_SRC_CYLON_OPS_JOIN_OP_HPP_
#define CYLON_SRC_CYLON_OPS_JOIN_OP_HPP_

#include "parallel_op.hpp"
#include "partition_op.hpp"

namespace cylon {

class JoinOpConfig {
 private:
  std::shared_ptr<PartitionOpConfig> partition_config;

 public:
  JoinOpConfig(std::shared_ptr<PartitionOpConfig> partition_config);

  std::shared_ptr<PartitionOpConfig> GetPartitionConfig();
};

class JoinOp : public Op {
 private:
  static const int JOIN_OP = 0;
  static const int LEFT = 2;
  static const int RIGHT = 3;

  static const int SHUFFLE = 4;
  static const int SORT = 5;
  static const int LOCAL_JOIN = 6;

 public:
  JoinOp(std::shared_ptr<CylonContext> ctx,
         std::shared_ptr<arrow::Schema> schema,
         std::shared_ptr<ResultsCallback> callback, std::shared_ptr<JoinOpConfig> config);

  bool Execute(int tag, std::shared_ptr<Table> table) override;

  void OnParentsFinalized() override;
};
}
#endif //CYLON_SRC_CYLON_OPS_JOIN_OP_HPP_

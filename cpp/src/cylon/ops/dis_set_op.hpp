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

#ifndef CYLON_SRC_CYLON_OPS_DIS_UNION_OP_HPP_
#define CYLON_SRC_CYLON_OPS_DIS_UNION_OP_HPP_

#include <cylon/ops/api/parallel_op.hpp>
#include <cylon/ops/kernels/set_kernel.hpp>

namespace cylon {

class DisSetOpConfig {
public:
  DisSetOpConfig(int leftSplits, int rightSplits) : left_splits(leftSplits), right_splits(rightSplits) {}

  int GetLeftSplits() const {
    return left_splits;
  }

  int GetRightSplits() const {
    return right_splits;
  }
private:
  int left_splits;
  int right_splits;
};

class DisSetOp : public RootOp {

 public:
  const static int32_t LEFT_RELATION = 100;
  const static int32_t RIGHT_RELATION = 200;

  DisSetOp(const std::shared_ptr<CylonContext> &ctx,
           const std::shared_ptr<arrow::Schema> &schema,
           int id,
           const ResultsCallback &callback,
           const DisSetOpConfig &config,
           cylon::kernel::SetOpType op_type);

  bool Execute(int tag, std::shared_ptr<Table> &table) override;
};
}
#endif //CYLON_SRC_CYLON_OPS_DIS_UNION_OP_HPP_

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


#ifndef CYLON_SRC_CYLON_OPS_KERNELS_UNION_HPP_
#define CYLON_SRC_CYLON_OPS_KERNELS_UNION_HPP_

#include <vector>
#include <unordered_set>
#include <queue>

#include <cylon/table.hpp>
#include <cylon/ops/kernels/row_comparator.hpp>

namespace cylon {
namespace kernel {

enum SetOpType {
  UNION,
  INTERSECT,
  SUBTRACT,
};

class SetOp {
 private:
 public:
  SetOp() {};
  virtual void InsertTable(int tag, const std::shared_ptr<cylon::Table> &table) = 0;
  virtual cylon::Status Finalize(std::shared_ptr<cylon::Table> &result) = 0;
};

std::unique_ptr<SetOp> CreateSetOp(const std::shared_ptr<CylonContext> &ctx,
                                      const std::shared_ptr<arrow::Schema> &schema,
                                      int64_t expected_rows,
                                      SetOpType op_type);

}
}
#endif //CYLON_SRC_CYLON_OPS_KERNELS_UNION_HPP_

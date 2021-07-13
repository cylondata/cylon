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

#include "table.hpp"
#include "ops.hpp"

namespace cylon {

Status JoinOperation(const std::shared_ptr <cylon::CylonContext> &ctx,
                         std::shared_ptr <cylon::Table> &left,
                         std::shared_ptr <cylon::Table> &right,
                         const cylon::join::config::JoinConfig &join_config,
                         std::shared_ptr <cylon::Table> &out) {
  const cylon::ResultsCallback &callback = [&](int tag, const std::shared_ptr <cylon::Table> &table) {
    out = table;
  };
  const auto &part_config = cylon::PartitionOpConfig(ctx->GetWorldSize(), {0});
  const auto &dist_join_config = cylon::DisJoinOpConfig(part_config, join_config);
  auto op = cylon::DisJoinOP(ctx, left->get_table()->schema(), 0, callback, dist_join_config);
  op.InsertTable(100, left);
  op.InsertTable(200, right);
  auto execution = op.GetExecution();
  execution->WaitForCompletion();
  return cylon::Status::OK();
}

Status UnionOperation(const std::shared_ptr <cylon::CylonContext> &ctx,
                          std::shared_ptr <cylon::Table> &left,
                          std::shared_ptr <cylon::Table> &right,
                          std::shared_ptr <cylon::Table> &out) {
  const cylon::ResultsCallback &callback = [&](int tag, const std::shared_ptr <cylon::Table> &table) {
    out = table;
  };
  cylon::DisUnionOpConfig unionOpConfig;
  auto op = cylon::DisSetOp(ctx, left->get_table()->schema(), 0, callback, unionOpConfig, cylon::kernel::UNION);
  op.InsertTable(100, left);
  op.InsertTable(200, right);
  auto execution = op.GetExecution();
  execution->WaitForCompletion();
  return cylon::Status::OK();
}

Status SubtractOperation(const std::shared_ptr <cylon::CylonContext> &ctx,
                             std::shared_ptr <cylon::Table> &left,
                             std::shared_ptr <cylon::Table> &right,
                             std::shared_ptr <cylon::Table> &out) {
  const cylon::ResultsCallback &callback = [&](int tag, const std::shared_ptr <cylon::Table> &table) {
    out = table;
  };
  cylon::DisUnionOpConfig unionOpConfig;
  auto op = cylon::DisSetOp(ctx, left->get_table()->schema(), 0, callback, unionOpConfig, cylon::kernel::SUBTRACT);
  op.InsertTable(100, left);
  op.InsertTable(200, right);
  auto execution = op.GetExecution();
  execution->WaitForCompletion();
  return cylon::Status::OK();
}

Status IntersectOperation(const std::shared_ptr <cylon::CylonContext> &ctx,
                              std::shared_ptr <cylon::Table> &left,
                              std::shared_ptr <cylon::Table> &right,
                              std::shared_ptr <cylon::Table> &out) {
  const cylon::ResultsCallback &callback = [&](int tag, const std::shared_ptr <cylon::Table> &table) {
    out = table;
  };
  cylon::DisUnionOpConfig unionOpConfig;
  auto op = cylon::DisSetOp(ctx, left->get_table()->schema(), 0, callback, unionOpConfig, cylon::kernel::INTERSECT);
  op.InsertTable(100, left);
  op.InsertTable(200, right);
  auto execution = op.GetExecution();
  execution->WaitForCompletion();
  return cylon::Status::OK();
}

} // namespace cylon

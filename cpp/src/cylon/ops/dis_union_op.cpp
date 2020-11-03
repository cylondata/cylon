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

#include "dis_union_op.hpp"
#include "partition_op.hpp"
#include "all_to_all_op.hpp"
#include "union_op.hpp"

cylon::DisUnionOp::DisUnionOp(const std::shared_ptr<CylonContext> &ctx,
                              const std::shared_ptr<arrow::Schema> &schema,
                              int id,
                              std::shared_ptr<ResultsCallback> &callback,
                              std::shared_ptr<DisUnionOpConfig> &config) : RootOp(ctx, schema, id, callback) {
  auto execution = new RoundRobinExecution();
  execution->AddOp(this);
  this->SetExecution(execution);

  const int32_t PARTITION_OP_ID = 0;
  const int32_t SHUFFLE_OP_ID = 1;
  const int32_t UNION_OP_ID = 2;

  // create graph
  std::vector<int> part_cols{};
  part_cols.reserve(schema->num_fields());
  for (int c = 0; c < schema->num_fields(); c++) {
    part_cols.push_back(c);
  }
  auto partition_op = new PartitionOp(ctx, schema, PARTITION_OP_ID, callback,
                                      std::make_shared<PartitionOpConfig>(ctx->GetWorldSize(),
                                                                          std::make_shared<std::vector<int>>(std::move(
                                                                              part_cols))));

  this->AddChild(partition_op);
  execution->AddOp(partition_op);

  auto shuffle_op = new AllToAllOp(ctx, schema, SHUFFLE_OP_ID, callback, std::make_shared<AllToAllOpConfig>());
  partition_op->AddChild(shuffle_op);
  execution->AddOp(shuffle_op);

  std::shared_ptr<UnionOpConfig> union_op_config = std::make_shared<UnionOpConfig>();
  auto union_op = new UnionOp(ctx, schema, UNION_OP_ID, callback, union_op_config);
  shuffle_op->AddChild(union_op);
  execution->AddOp(union_op);
  // done creating graph


}



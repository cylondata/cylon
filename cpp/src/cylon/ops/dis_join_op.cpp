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
#include "dis_join_op.hpp"

#include <utility>
#include "all_to_all_op.hpp"
#include "merge_op.hpp"

cylon::DisJoinOP::DisJoinOP(std::shared_ptr<cylon::CylonContext> ctx,
                            std::shared_ptr<arrow::Schema> schema,
                            int id,
                            std::shared_ptr<ResultsCallback> callback,
                            std::shared_ptr<DisJoinOpConfig> config) :
                                              RootOp(ctx, schema, id, callback) {
  auto execution = new RoundRobinExecution();
  execution->AddOp(this);
  this->SetExecution(execution);

  this->config = std::move(config);
  const std::vector<int32_t> PARTITION_IDS = {LEFT_RELATION, RIGHT_RELATION};
  const int32_t SHUFFLE_OP_ID = 3;
  const int32_t JOIN_OP_ID = 4;
  const int32_t MERGE_OP_ID = 5;

  // create graph
  std::vector<int> part_cols = {this->config->GetJoinConfig()->GetLeftColumnIdx()};
  auto join_op = new JoinOp(ctx, schema, JOIN_OP_ID, callback, this->config->GetJoinConfig());
  auto left_shuffle_op = new AllToAllOp(ctx, schema, SHUFFLE_OP_ID, callback,
                                        std::make_shared<AllToAllOpConfig>());

  auto left_partition_op = new PartitionOp(ctx, schema, PARTITION_IDS[0], callback,
                                      std::make_shared<PartitionOpConfig>(ctx->GetWorldSize(),
                                              std::make_shared<std::vector<int>>(part_cols)));
  this->AddChild(left_partition_op);
  execution->AddOp(left_partition_op);

  left_partition_op->AddChild(left_shuffle_op);
  execution->AddOp(left_shuffle_op);
  // sorting op can be added here
  auto left_merge_op = new MergeOp(ctx, schema, MERGE_OP_ID, callback);
  left_shuffle_op->AddChild(left_merge_op);
  execution->AddOp(left_merge_op);
  left_merge_op->AddChild(join_op);

  auto right_shuffle_op = new AllToAllOp(ctx, schema, SHUFFLE_OP_ID, callback,
                                         std::make_shared<AllToAllOpConfig>());

  auto right_partition_op = new PartitionOp(ctx, schema, PARTITION_IDS[0], callback,
                                           std::make_shared<PartitionOpConfig>(ctx->GetWorldSize(),
                                           std::make_shared<std::vector<int>>(part_cols)));
  this->AddChild(right_partition_op);
  execution->AddOp(right_partition_op);
  right_partition_op->AddChild(right_shuffle_op);
  execution->AddOp(right_shuffle_op);
  // sorting op can be added here
  auto right_merge_op = new MergeOp(ctx, schema, MERGE_OP_ID, callback);
  right_shuffle_op->AddChild(right_merge_op);
  execution->AddOp(right_merge_op);
  left_merge_op->AddChild(join_op);
  execution->AddOp(join_op);
}
bool cylon::DisJoinOP::Execute(int tag, shared_ptr<Table> table) {
  if (tag != LEFT_RELATION && tag != RIGHT_RELATION) {
    LOG(INFO) << "Unknown tag";
    return false;
  }
  LOG(INFO) << "Join op";
  this->InsertTable(tag, table);
  return true;
}

void cylon::DisJoinOP::OnParentsFinalized() {
  // do nothing
}

bool cylon::DisJoinOP::Finalize() {
  return true;
}

std::shared_ptr<cylon::PartitionOpConfig> cylon::DisJoinOpConfig::GetPartitionConfig() {
  return this->partition_config;
}

cylon::DisJoinOpConfig::DisJoinOpConfig(std::shared_ptr<PartitionOpConfig> partition_config,
                  std::shared_ptr<cylon::join::config::JoinConfig> join_config) {
  this->partition_config = std::move(partition_config);
  this->join_config = std::move(join_config);
}

const std::shared_ptr<cylon::join::config::JoinConfig> &
    cylon::DisJoinOpConfig::GetJoinConfig() const {
  return join_config;
}

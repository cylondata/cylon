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
#include "split_op.hpp"

cylon::DisJoinOP::DisJoinOP(const std::shared_ptr<cylon::CylonContext>& ctx,
                            const std::shared_ptr<arrow::Schema> &schema,
                            int id,
                            const std::shared_ptr<ResultsCallback> &callback,
                            const std::shared_ptr<DisJoinOpConfig> &config) :
    RootOp(ctx, schema, id, callback) {
  auto execution = new RoundRobinExecution();
  execution->AddOp(this);
  this->SetExecution(execution);

  this->config = config;
  const std::vector<int32_t> PARTITION_IDS = {LEFT_RELATION, RIGHT_RELATION};
  const int32_t JOIN_OP_ID = 4;
//  const int32_t MERGE_OP_ID = 5;
  // create graph
  // local join
  auto join_op = new JoinOp(ctx, schema, JOIN_OP_ID, callback, this->config->GetJoinConfig());

  for (int32_t relation_id:PARTITION_IDS) {
    std::vector<int> part_cols = {this->config->GetJoinConfig()->GetLeftColumnIdx()};
    auto partition_op = new PartitionOp(ctx, schema, relation_id, callback,
                                        std::make_shared<PartitionOpConfig>(ctx->GetWorldSize(),
                                        std::make_shared<std::vector<int>>(part_cols)));
    this->AddChild(partition_op);
    execution->AddOp(partition_op);
    auto shuffle_op = new AllToAllOp(ctx, schema, relation_id, callback,
                                     std::make_shared<AllToAllOpConfig>());
    partition_op->AddChild(shuffle_op);
    execution->AddOp(shuffle_op);
//    auto merge_op = new MergeOp(ctx, schema, MERGE_OP_ID, callback);
    std::shared_ptr<SplitOpConfig> kPtr = SplitOpConfig::Make(2000, {0});
    LOG(INFO) << "aaaaa";
    auto split_op = new SplitOp(ctx, schema, relation_id, callback, kPtr);
    shuffle_op->AddChild(split_op);
    execution->AddOp(split_op);
    split_op->AddChild(join_op);
  }
  execution->AddOp(join_op);
}

bool cylon::DisJoinOP::Execute(int tag, shared_ptr<Table> table) {
  if (tag != LEFT_RELATION && tag != RIGHT_RELATION) {
    LOG(INFO) << "Unknown tag";
    return false;
  }
  this->InsertToChild(tag, tag, table);
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

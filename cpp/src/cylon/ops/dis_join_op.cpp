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
#include "all_to_all_op.hpp"
cylon::DisJoinOP::DisJoinOP(std::shared_ptr<cylon::CylonContext> ctx,
                            std::shared_ptr<arrow::Schema> schema,
                            int id,
                            shared_ptr<ResultsCallback> callback,
                            shared_ptr<DisJoinOpConfig> config) : Op(ctx, schema, id, callback) {
  this->config = config;

  const std::vector<int32_t> PARTITION_IDS = {LEFT_RELATION, RIGHT_RELATION};
  const int32_t SHUFFLE_OP_ID = 3;
  const int32_t JOIN_OP_ID = 4;

  // create graph

  std::vector<int> part_cols = {this->config->GetJoinConfig()->GetJoinColumn()};

  auto join_op = new JoinOp(ctx, schema, JOIN_OP_ID, callback, this->config->GetJoinConfig());

  for (int32_t relation = 0; relation < 2; relation++) {
    auto partition_op = new PartitionOp(ctx, schema, PARTITION_IDS[relation], callback,
                                        std::make_shared<PartitionOpConfig>(ctx->GetWorldSize(),
                                                                            std::make_shared<std::vector<int>>(part_cols)));
    this->AddChild(partition_op);
    auto shuffle_op = new AllToAllOp(ctx, schema, SHUFFLE_OP_ID, callback, std::make_shared<AllToAllOpConfig>());
    partition_op->AddChild(shuffle_op);

    // sorting op can be added here

    shuffle_op->AddChild(join_op);
  }
}
bool cylon::DisJoinOP::Execute(int tag, shared_ptr<Table> table) {
  if (tag != LEFT_RELATION || tag != RIGHT_RELATION) {
    LOG(INFO) << "Unknown tag";
    return false;
  }
  this->InsertTable(tag, table);
  return true;
}

void cylon::DisJoinOP::OnParentsFinalized() {
  // do nothing
}

bool cylon::DisJoinOP::Finalize() {
  return true;
}

shared_ptr<cylon::PartitionOpConfig> cylon::DisJoinOpConfig::GetPartitionConfig() {
  return this->partition_config;
}

cylon::DisJoinOpConfig::DisJoinOpConfig(shared_ptr<PartitionOpConfig> partition_config,
                                        shared_ptr<JoinOpConfig> join_config) {
  this->partition_config = partition_config;
  this->join_config = join_config;
}

const shared_ptr<cylon::JoinOpConfig> &cylon::DisJoinOpConfig::GetJoinConfig() const {
  return join_config;
}

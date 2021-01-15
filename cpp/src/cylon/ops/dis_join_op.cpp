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

cylon::DisJoinOP::DisJoinOP(const std::shared_ptr<CylonContext> &ctx,
                            const std::shared_ptr<arrow::Schema> &schema,
                            int id,
                            const ResultsCallback &callback,
                            const DisJoinOpConfig &config) : RootOp(ctx, schema, id, callback) {
  auto execution = new JoinExecution();
  execution->AddP(this);
  this->SetExecution(execution);

  const std::vector<int32_t> PARTITION_IDS = {LEFT_RELATION, RIGHT_RELATION};
  const int32_t JOIN_OP_ID = 4;
//  const int32_t MERGE_OP_ID = 5;
  // create graph
  // local join
  auto join_op = new JoinOp(ctx, schema, JOIN_OP_ID, callback, config.join_config);

  for (int32_t relation_id:PARTITION_IDS) {
    // todo in here, LEFT and RIGH col_idx should be equal
    auto partition_op = new PartitionOp(ctx, schema, relation_id, callback,
                                        {ctx->GetWorldSize(), {config.join_config.GetLeftColumnIdx()}});
    this->AddChild(partition_op);
    if (relation_id == LEFT_RELATION) {
      execution->AddP(partition_op);
    } else {
      execution->AddS(partition_op);
    }
    auto shuffle_op = new AllToAllOp(ctx, schema, relation_id, callback, {});
    partition_op->AddChild(shuffle_op);
    if (relation_id == LEFT_RELATION) {
      execution->AddP(shuffle_op);
    } else {
      execution->AddS(shuffle_op);
    }
//    auto merge_op = new MergeOp(ctx, schema, MERGE_OP_ID, callback);
    SplitOpConfig kPtr{100, {0}};
    auto split_op = new SplitOp(ctx, schema, relation_id, callback, kPtr);
    shuffle_op->AddChild(split_op);
    if (relation_id == LEFT_RELATION) {
      execution->AddP(split_op);
    } else {
      execution->AddS(split_op);
    }
    split_op->AddChild(join_op);
  }
  execution->AddJoin(join_op);
}

bool cylon::DisJoinOP::Execute(int tag, std::shared_ptr<Table> &table) {
  if (tag != LEFT_RELATION && tag != RIGHT_RELATION) {
    LOG(INFO) << "Unknown tag";
    return false;
  }
  this->InsertToChild(tag, tag, table);
  return true;
}


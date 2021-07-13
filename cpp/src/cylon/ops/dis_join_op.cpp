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
#include "split_op.hpp"

cylon::DisJoinOP::DisJoinOP(const std::shared_ptr<CylonContext> &ctx,
                            const std::shared_ptr<arrow::Schema> &schema,
                            int id,
                            const ResultsCallback &callback,
                            const DisJoinOpConfig &config) : RootOp(ctx, schema, id, callback) {
  auto execution = new ForkJoinExecution();
  execution->AddLeft(this);
  this->SetExecution(execution);

  const std::vector<int32_t> PARTITION_IDS = {LEFT_RELATION, RIGHT_RELATION};
  const int32_t JOIN_OP_ID = 4;
//  const int32_t MERGE_OP_ID = 5;
  // create graph
  Op *partition_op, *shuffle_op, *split_op, *join_op;

  // build left sub tree
  partition_op = new PartitionOp(ctx, schema, LEFT_RELATION, callback,
                                 {ctx->GetWorldSize(), {config.join_config.GetLeftColumnIdx()}});
  this->AddChild(partition_op);
  execution->AddLeft(partition_op);

  shuffle_op = new AllToAllOp(ctx, schema, LEFT_RELATION, callback, {});
  partition_op->AddChild(shuffle_op);
  execution->AddLeft(shuffle_op);

  split_op = new SplitOp(ctx, schema, LEFT_RELATION, callback, {8000, {config.join_config.GetLeftColumnIdx()}});
  shuffle_op->AddChild(split_op);
  execution->AddLeft(split_op);

  // add join op
  join_op = new JoinOp(ctx, schema, JOIN_OP_ID, callback, config.join_config);
  split_op->AddChild(join_op);
  execution->AddFinal(join_op);


  // build right sub tree
  partition_op = new PartitionOp(ctx, schema, RIGHT_RELATION, callback,
                                 {ctx->GetWorldSize(), {config.join_config.GetLeftColumnIdx()}});
  this->AddChild(partition_op);
  execution->AddRight(partition_op);

  shuffle_op = new AllToAllOp(ctx, schema, RIGHT_RELATION, callback, {});
  partition_op->AddChild(shuffle_op);
  execution->AddRight(shuffle_op);

  split_op = new SplitOp(ctx, schema, RIGHT_RELATION, callback, {8000, {config.join_config.GetRightColumnIdx()}});
  shuffle_op->AddChild(split_op);
  execution->AddRight(split_op);

  split_op->AddChild(join_op); // join_op is already initialized
}

bool cylon::DisJoinOP::Execute(int tag, std::shared_ptr<Table> &table) {
  if (tag != LEFT_RELATION && tag != RIGHT_RELATION) {
    LOG(INFO) << "Unknown tag";
    return false;
  }
  this->InsertToChild(tag, tag, table);
  return true;
}




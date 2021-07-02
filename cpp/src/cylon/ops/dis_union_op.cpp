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
#include "split_op.hpp"

cylon::DisUnionOp::DisUnionOp(const std::shared_ptr<CylonContext> &ctx,
                              const std::shared_ptr<arrow::Schema> &schema,
                              int id,
                              const ResultsCallback &callback,
                              const DisUnionOpConfig &config)
    : RootOp(ctx, schema, id, callback) {
  auto execution = new JoinExecution();
  execution->AddP(this);
  this->SetExecution(execution);

  const int32_t UNION_OP_ID = 4;

  // create graph
  std::vector<int> part_cols(schema->num_fields());
  std::iota(part_cols.begin(), part_cols.end(), 0);

  const std::vector<int32_t> PARTITION_IDS = {LEFT_RELATION, RIGHT_RELATION};
  // create graph
  Op *partition_op, *shuffle_op, *split_op, *union_op;

  // build left sub tree
  partition_op = new PartitionOp(ctx, schema, LEFT_RELATION, callback,
                                 {ctx->GetWorldSize(), std::move(part_cols)});
  this->AddChild(partition_op);
  execution->AddP(partition_op);

  shuffle_op = new AllToAllOp(ctx, schema, LEFT_RELATION, callback, {});
  partition_op->AddChild(shuffle_op);
  execution->AddP(shuffle_op);

  split_op = new SplitOp(ctx, schema, LEFT_RELATION, callback, {8000, {0}});
  shuffle_op->AddChild(split_op);
  execution->AddP(split_op);

  // add join op
  UnionOpConfig union_config;
  union_op = new UnionOp(ctx, schema, UNION_OP_ID, callback, union_config);
  split_op->AddChild(union_op);
  execution->AddP(union_op);

  // build right sub tree
  partition_op = new PartitionOp(ctx, schema, RIGHT_RELATION, callback,
                                 {ctx->GetWorldSize(), std::move(part_cols)});
  this->AddChild(partition_op);
  execution->AddS(partition_op);

  shuffle_op = new AllToAllOp(ctx, schema, RIGHT_RELATION, callback, {});
  partition_op->AddChild(shuffle_op);
  execution->AddS(shuffle_op);

  split_op = new SplitOp(ctx, schema, RIGHT_RELATION, callback, {8000, {0}});
  shuffle_op->AddChild(split_op);
  execution->AddS(split_op);

  split_op->AddChild(union_op); // join_op is already initialized
}

bool cylon::DisUnionOp::Execute(int tag, std::shared_ptr<Table> &table) {
  if (tag != LEFT_RELATION && tag != RIGHT_RELATION) {
    LOG(INFO) << "Unknown tag";
    return false;
  }
  this->InsertToChild(tag, tag, table);
  return true;
}


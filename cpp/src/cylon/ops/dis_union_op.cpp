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

cylon::DisUnionOp::DisUnionOp(std::shared_ptr<cylon::CylonContext> ctx,
                              std::shared_ptr<arrow::Schema> schema,
                              int id,
                              shared_ptr<ResultsCallback> callback,
                              shared_ptr<DisUnionOpConfig> config) : Op(ctx, schema, id, callback, true) {
  const int32_t PARTITION_OP_ID = 0;
  const int32_t SHUFFLE_OP_ID = 1;
  const int32_t UNION_OP_ID = 2;

  // create graph
  std::vector<int> part_cols{};
  for (int c = 0; c < schema->num_fields(); c++) {
    part_cols.push_back(c);
  }
  auto partition_op = new PartitionOp(ctx, schema, PARTITION_OP_ID,  callback,
                                      std::make_shared<PartitionOpConfig>(ctx->GetWorldSize(),
                                                                          std::make_shared<std::vector<int>>(part_cols)));

  this->AddChild(partition_op);

  auto shuffle_op = new AllToAllOp(ctx, schema, SHUFFLE_OP_ID, callback, std::make_shared<AllToAllOpConfig>());
  partition_op->AddChild(shuffle_op);

  auto union_op = new UnionOp(ctx, schema, UNION_OP_ID,  callback, std::make_shared<UnionOpConfig>());
  shuffle_op->AddChild(union_op);

  // done creating graph
}

bool cylon::DisUnionOp::Execute(int tag, shared_ptr<Table> table) {
  // todo do slicing based on data size
  this->InsertToAllChildren(tag, table);
  return true;
}

void cylon::DisUnionOp::OnParentsFinalized() {
  // do nothing
}

bool cylon::DisUnionOp::Finalize() {
  return true;
}

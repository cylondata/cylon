#include "dis_union_op.h"
#include "partition_op.hpp"
#include "shuffle_op.h"
#include "union_op.hpp"

cylon::DisUnionOp::DisUnionOp(std::shared_ptr<cylon::CylonContext> ctx,
                              std::shared_ptr<arrow::Schema> schema,
                              int id,
                              std::function<int(int)> router,
                              shared_ptr<ResultsCallback> callback,
                              shared_ptr<DisUnionOpConfig> config) : Op(ctx, schema, id, router, callback, true) {
  const int32_t PARTITION_OP_ID = 0;
  const int32_t SHUFFLE_OP_ID = 1;
  const int32_t UNION_OP_ID = 2;

  // create graph
  std::vector<int> part_cols{};
  for (int c = 0; c < schema->num_fields(); c++) {
    part_cols.push_back(c);
  }
  auto partition_op = new PartitionOp(ctx, schema, PARTITION_OP_ID, router, callback,
                                      std::make_shared<PartitionOpConfig>(ctx->GetWorldSize(),
                                                                          std::make_shared<std::vector<int>>(part_cols)));

  this->AddChild(partition_op);

  auto shuffle_op = new ShuffleOp(ctx, schema, SHUFFLE_OP_ID, router, callback, std::make_shared<ShuffleOpConfig>());
  partition_op->AddChild(shuffle_op);

  auto union_op = new UnionOp(ctx, schema, UNION_OP_ID, router, callback, std::make_shared<UnionOpConfig>());
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

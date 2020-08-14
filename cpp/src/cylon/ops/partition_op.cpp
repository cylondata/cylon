#include "partition_op.h"
#include <ops/kernels/partition.h>

cylon::PartitionOp::PartitionOp(int id, std::function<int(int)> router,
                                shared_ptr<ResultsCallback> callback) : Op(id, router, callback) {}

void cylon::PartitionOp::init(std::shared_ptr<CylonContext> ctx, std::shared_ptr<OpConfig> op_config) {
  Op::init(ctx, op_config);
  this->ctx_ = ctx;
  this->no_of_partitions = ctx->GetWorldSize();
}

cylon::PartitionOp::PartitionOp(std::function<int(int)> router,
                                shared_ptr<ResultsCallback> callback) : Op(PARTITION_OP, router, callback) {}

void cylon::PartitionOp::execute(int tag, std::shared_ptr<Table> table) {
  std::unordered_map<int, std::shared_ptr<Table>> out;

  // todo pass ctx as a shared pointer
  cylon::kernel::HashPartition(&*this->ctx_, table, this->hash_columns,
                               this->no_of_partitions, &out);
  for (auto const &tab:out) {
    this->InsertToAllChildren(tab.first, tab.second);
  }
}

bool cylon::PartitionOp::ready() {
  return true;
}
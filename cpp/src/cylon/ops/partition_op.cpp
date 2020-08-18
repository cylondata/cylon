#include "partition_op.h"
#include <ops/kernels/computation/partition.h>

cylon::PartitionOp::PartitionOp(std::shared_ptr<cylon::CylonContext> ctx,
                                int id, std::function<int(int)> router,
                                std::shared_ptr<ResultsCallback> callback,
                                std::shared_ptr<PartitionOpConfig> config) : Op(ctx, id, router, callback) {}

void cylon::PartitionOp::execute(int tag, std::shared_ptr<Table> table) {
  std::unordered_map<int, std::shared_ptr<Table>> out;

  // todo pass ctx as a shared pointer
  cylon::kernel::HashPartition(&*this->ctx_, table, *this->config->HashColumns(),
                               this->config->NoOfPartitions(), &out);
  for (auto const &tab:out) {
    this->InsertToAllChildren(tab.first, tab.second);
  }
}

cylon::PartitionOpConfig::PartitionOpConfig(int no_of_partitions,
                                            std::shared_ptr<std::vector<int>> hash_columns)
    : no_of_partitions(no_of_partitions),
      hash_columns(hash_columns) {

}

int cylon::PartitionOpConfig::NoOfPartitions() {
  return this->no_of_partitions;
}

std::shared_ptr<std::vector<int>> cylon::PartitionOpConfig::HashColumns() {
  return this->hash_columns;
}

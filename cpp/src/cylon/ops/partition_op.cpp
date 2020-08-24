#include "partition_op.hpp"
#include <ops/kernels/partition.hpp>

cylon::PartitionOp::PartitionOp(std::shared_ptr<cylon::CylonContext> ctx,
                                std::shared_ptr<arrow::Schema> schema,
                                int id,
                                std::shared_ptr<ResultsCallback> callback,
                                std::shared_ptr<PartitionOpConfig> config) : Op(ctx, schema, id, callback),
                                                                             config(config) {}

bool cylon::PartitionOp::Execute(int tag, std::shared_ptr<Table> table) {

  LOG(INFO) << "Executing partition op";

  std::unordered_map<int, std::shared_ptr<Table>> out;

  // todo pass ctx as a shared pointer
  cylon::kernel::HashPartition(&*this->ctx_, table, *this->config->HashColumns(),
                               this->config->NoOfPartitions(), &out);
  for (auto const &tab:out) {
    LOG(INFO) << "PARTITION OP : sending to " << tab.first;
    this->InsertToAllChildren(tab.first, tab.second);
  }
  return true;
}

bool cylon::PartitionOp::Finalize() {
  return true;
}

void cylon::PartitionOp::OnParentsFinalized() {
  // do nothing
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

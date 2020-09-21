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

#include "partition_op.hpp"
#include <ops/kernels/partition.hpp>

cylon::PartitionOp::PartitionOp(const std::shared_ptr<cylon::CylonContext> &ctx,
                                const std::shared_ptr<arrow::Schema> &schema,
                                int id,
                                const std::shared_ptr<ResultsCallback> &callback,
                                const std::shared_ptr<PartitionOpConfig> &config) :
                                Op(ctx, schema, id, callback), config(config) {}

bool cylon::PartitionOp::Execute(int tag, std::shared_ptr<Table> table) {
  if (!started_time) {
    start = std::chrono::high_resolution_clock::now();
    started_time = true;
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  std::unordered_map<int, std::shared_ptr<Table>> out;
  // todo pass ctx as a shared pointer
  std::shared_ptr<std::vector<int>> kPtr = this->config->HashColumns();
  if (kPtr->size() > 1) {
    cylon::kernel::HashPartition(&*this->ctx_, table, *kPtr,
                                   this->config->NoOfPartitions(), &out);
  } else {
    cylon::kernel::HashPartition(&*this->ctx_, table, kPtr->at(0),
                                 this->config->NoOfPartitions(), &out);
  }
  for (auto const &tab:out) {
    this->InsertToAllChildren(tab.first, tab.second);
  }

  // we are going to free if retain is set to false
  if (!table->IsRetain()) {
    table.reset();
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  exec_time += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  return true;
}

bool cylon::PartitionOp::Finalize() {
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Partition time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - start).count();
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

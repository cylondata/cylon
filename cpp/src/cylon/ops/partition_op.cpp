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

#include <glog/logging.h>
#include <cylon/partition/partition.hpp>
#include <cylon/ops/partition_op.hpp>

#include <iomanip>

cylon::PartitionOp::PartitionOp(const std::shared_ptr<cylon::CylonContext> &ctx,
                                const std::shared_ptr<arrow::Schema> &schema,
                                int id,
                                const ResultsCallback &callback,
                                const PartitionOpConfig &config)
    : Op(ctx, schema, id, callback), config(config) {}

bool cylon::PartitionOp::Execute(int tag, std::shared_ptr<Table> &table) {
  if (!started_time) {
    start = std::chrono::high_resolution_clock::now();
    started_time = true;
  }
  std::vector<std::shared_ptr<cylon::Table>> out;
  const auto &status = PartitionByHashing(table, config.hash_columns, config.num_partitions, out);
  if (!status.is_ok()) {
    LOG(ERROR) << "hash partition failed: " << status.get_msg();
    return false;
  }

  for (int i = 0; i < config.num_partitions; i++) {
    this->InsertToAllChildren(i, out[i]);
#if CYLON_DEBUG
    sleep(ctx_->GetRank());
    std::cout << "****from " << ctx_->GetRank() << " to " << i << std::endl;
    out[i]->Print();
    std::cout << "-------------------------------" << std::endl;
#endif
  }
  out.clear();

  // we are going to free if retain is set to false
  if (!table->IsRetain()) {
    table.reset();
  }

  return true;
}

bool cylon::PartitionOp::Finalize() {
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << ctx_->GetRank() << " Partition time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - start).count();
  return true;
}

void cylon::PartitionOp::OnParentsFinalized() {
  // do nothing
}




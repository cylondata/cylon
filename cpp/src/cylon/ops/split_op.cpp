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

#include <vector>
#include <chrono>
#include <memory>

#include <glog/logging.h>
#include <cylon/ops/split_op.hpp>
#include <cylon/ctx/arrow_memory_pool_utils.hpp>
#include <cylon/arrow/arrow_partition_kernels.hpp>
#include <cylon/status.hpp>
#include <cylon/table.hpp>

namespace cylon {

cylon::SplitOp::SplitOp(const std::shared_ptr<CylonContext> &ctx,
                        const std::shared_ptr<arrow::Schema> &schema,
                        int32_t id,
                        const ResultsCallback &callback,
                        const SplitOpConfig &cfg)
    : Op(ctx, schema, id, callback), partition_kernel(ctx, schema, cfg.num_partitions, cfg.hash_columns) {
}

bool cylon::SplitOp::Execute(int tag, std::shared_ptr<Table> &cy_table) {
  if (!started_time) {
    start = std::chrono::high_resolution_clock::now();
    started_time = true;
  }
  count++;
  auto t1 = std::chrono::high_resolution_clock::now();
  const Status &status = partition_kernel.Process(tag, cy_table);
  if (!status.is_ok()) {
    LOG(ERROR) << "hash partition failed: " << status.get_msg();
    return false;
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  exec_time += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  return true;
}

void cylon::SplitOp::OnParentsFinalized() {

}

bool cylon::SplitOp::Finalize() {
  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<std::shared_ptr<Table>> partitions;
  const Status &status = partition_kernel.Finish(partitions);
  if (!status.is_ok()) {
    LOG(ERROR) << "split finish failed: " << status.get_msg();
    return false;
  }

  for (auto & partition : partitions) {
    partition->retainMemory(false);
    InsertToAllChildren(id, partition);
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Split time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - start).count()
            << " Fin time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
            << " Split time: " << exec_time
            << " Call count: " << count;
  return true;
}

}


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

#include "ops/split_op.hpp"
#include <ctx/arrow_memory_pool_utils.hpp>
#include <arrow/arrow_partition_kernels.hpp>
#include <status.hpp>
#include <utility>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <memory>
#include <table.hpp>
#include <partition/partition.hpp>

namespace cylon {

cylon::SplitOp::SplitOp(const std::shared_ptr<CylonContext> &ctx,
                        const std::shared_ptr<arrow::Schema> &schema,
                        int32_t id,
                        const ResultsCallback &callback,
                        const SplitOpConfig &cfg)
    : Op(ctx, schema, id, callback), partition_kernel(ctx, schema, cfg.num_partitions, cfg.hash_columns) {
  /*int kSize = config_->NoOfPartitions();
  targets.resize(kSize);
  hash_targets.resize(kSize * ctx_->GetWorldSize());
  std::iota(std::begin(targets), std::end(targets), 0);
  std::iota(std::begin(hash_targets), std::end(hash_targets), 0);

  split_kernels.reserve(schema->num_fields());
  for (int i = 0; i < schema->num_fields(); i++) {
    split_kernels.push_back(CreateStreamingSplitter(schema->fields()[i]->type(),
                                                    targets.size(),
                                                    cylon::ToArrowPool(ctx)));
  }*/
}

bool cylon::SplitOp::Execute(int tag, std::shared_ptr<Table> &cy_table) {
  if (!started_time) {
    start = std::chrono::high_resolution_clock::now();
    started_time = true;
  }
  count++;
  auto t1 = std::chrono::high_resolution_clock::now();
/*  std::shared_ptr<arrow::Table> table = cy_table->get_table();
  // first we need to calculate the hash
  std::shared_ptr<arrow::Array> arr = table->column(config_->HashColumns()[0])->chunk(0);
  int64_t length = arr->length();
  size_t size = ctx_->GetWorldSize();
  size_t kI = config_->NoOfPartitions();
  std::vector<uint32_t> cnts(kI * size, 0);
  std::vector<uint32_t> outPartitions;
  outPartitions.reserve(length);
  Status status = HashPartitionArray(cylon::ToArrowPool(ctx_), arr,
                                     hash_targets, &outPartitions, cnts);
  std::string s = " ";
  for (int i = 0; i < length; i++) {
    outPartitions[i] = outPartitions[i] / size;
  }

  int first = 0;
  for (size_t i = 0; i < kI; i++) {
    if (cnts[i] != 0) {
      first = i;
      break;
    }
  }
  
  for (size_t i = 0; i < kI; i++) {
//    LOG(INFO) << "Counts i=" << i << " size=" << size << " cnts[i]=" << cnts[i] << " cnts[i *size]=" << cnts[i * size];
    cnts[i] = cnts[i * size + first];
  }
  // now split
  for (int i = 0; i < table->num_columns(); i++) {
    std::shared_ptr<arrow::DataType> type = table->column(i)->chunk(0)->type();
    std::shared_ptr<arrow::Array> array = table->column(i)->chunk(0);

    std::shared_ptr<StreamingSplitKernel> splitKernel = split_kernels[i];
    splitKernel->Split(array, outPartitions, cnts);
  }*/
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
/*  // keep arrays for each target, these arrays are used for creating the table
  std::unordered_map<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>> data_arrays;
  int kI = config_->NoOfPartitions();
  for (int t = 0; t < kI; t++) {
    data_arrays.insert(
        std::pair<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>>(
            t, std::make_shared<std::vector<std::shared_ptr<arrow::Array>>>()));
  }
  
  // now we can finish the streaming kernels
  for (int i = 0; i < schema->num_fields(); i++) {
    std::shared_ptr<StreamingSplitKernel> splitKernel = split_kernels[i];
    std::unordered_map<int, std::shared_ptr<arrow::Array>> splited_arrays;
    splitKernel->Finish(splited_arrays);

    for (const auto &x : splited_arrays) {
      std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>> cols = data_arrays[x.first];
      cols->push_back(x.second);
    }
  }

  for (int i = 0; i < kI; i++) {
    std::shared_ptr<arrow::Table> t = arrow::Table::Make(schema, *data_arrays[i]);
    std::shared_ptr<cylon::Table> kY = std::make_shared<cylon::Table>(t, ctx_);
    kY->retainMemory(false);
    this->InsertToAllChildren(id, kY);
  }

  this->split_kernels.clear();*/

  std::vector<std::shared_ptr<Table>> partitions;
  const Status &status = partition_kernel.Finish(partitions);
  if (!status.is_ok()) {
    LOG(ERROR) << "split finish failed: " << status.get_msg();
    return false;
  }

  for (size_t i = 0; i < partitions.size(); i++) {
    partitions[i]->retainMemory(false);
    InsertToAllChildren(i, partitions[i]);
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


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
#include <unordered_map>
#include <memory>
#include <table.hpp>

namespace cylon {

Status HashPartition(CylonContext *ctx, const std::shared_ptr<cylon::Table> &cy_table,
                     int hash_column, int no_of_partitions,
                     std::unordered_map<int, std::shared_ptr<cylon::Table>> *out) {
  std::shared_ptr<arrow::Table> table = cy_table->get_table();
  // keep arrays for each target, these arrays are used for creating the table
  std::unordered_map<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>> data_arrays;
  std::vector<int> partitions;
  for (int t = 0; t < no_of_partitions; t++) {
    partitions.push_back(t);
    data_arrays.insert(
        std::pair<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>>(
            t, std::make_shared<std::vector<std::shared_ptr<arrow::Array>>>()));
  }
  std::shared_ptr<arrow::Array> arr = table->column(hash_column)->chunk(0);
  int64_t length = arr->length();

  auto t1 = std::chrono::high_resolution_clock::now();
  // first we partition the table
  std::vector<int64_t> outPartitions;
  outPartitions.reserve(length);
  std::vector<uint32_t> counts(no_of_partitions, 0);
  Status status = HashPartitionArray(cylon::ToArrowPool(ctx), arr,
                                     partitions, &outPartitions, counts);
  if (!status.is_ok()) {
    LOG(FATAL) << "Failed to create the hash partition";
    return status;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Calculating hash time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  for (int i = 0; i < table->num_columns(); i++) {
    std::shared_ptr<arrow::DataType> type = table->column(i)->chunk(0)->type();
    std::shared_ptr<arrow::Array> array = table->column(i)->chunk(0);

    std::shared_ptr<ArrowArraySplitKernel> splitKernel;
    status = CreateSplitter(type, cylon::ToArrowPool(ctx), &splitKernel);
    if (!status.is_ok()) {
      LOG(FATAL) << "Failed to create the splitter";
      return status;
    }
    // this one outputs arrays for each target as a map
    std::unordered_map<int, std::shared_ptr<arrow::Array>> splited_arrays;
    splitKernel->Split(array, outPartitions, partitions, splited_arrays, counts);
    for (const auto &x : splited_arrays) {
      std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>> cols = data_arrays[x.first];
      cols->push_back(x.second);
    }
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Building hashed tables time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
  // now insert these array to
  for (const auto &x : data_arrays) {
    std::shared_ptr<arrow::Table> t = arrow::Table::Make(table->schema(), *x.second);
    std::shared_ptr<cylon::Table> kY = std::make_shared<cylon::Table>(t, ctx);
    out->insert(std::pair<int, std::shared_ptr<cylon::Table>>(x.first, kY));
  }
  return Status::OK();
}

cylon::SplitOp::SplitOp(const std::shared_ptr<CylonContext> &ctx,
                        const std::shared_ptr<arrow::Schema> &schema,
                        int32_t id,
                        const std::shared_ptr<ResultsCallback> &callback,
                        const std::shared_ptr<SplitOpConfig> &cfg) :
    Op(ctx, schema, id, callback) {
  config_ = cfg;
  hash_column_ = (*(config_->HashColumns()))[0];
  int kSize = config_->NoOfPartitions();
  targets.resize(kSize);
  hash_targets.resize(kSize * ctx_->GetWorldSize());
  std::iota (std::begin(targets), std::end(targets), 0);
  std::iota (std::begin(hash_targets), std::end(hash_targets), 0);
  for (int i = 0; i < schema->num_fields(); i++) {
    std::shared_ptr<ArrowArrayStreamingSplitKernel> splitKernel;
    Status status = CreateStreamingSplitter(schema->fields()[i]->type(), targets,
        cylon::ToArrowPool(ctx.get()), &splitKernel);
    received_tables_.push_back(splitKernel);
  }
}

bool cylon::SplitOp::Execute(int tag, std::shared_ptr<Table> cy_table) {
  std::shared_ptr<arrow::Table> table = cy_table->get_table();
  // first we need to calculate the hash
  std::shared_ptr<arrow::Array> arr = table->column(hash_column_)->chunk(0);
  int64_t length = arr->length();
  int size = ctx_->GetWorldSize();
  std::vector<uint32_t> cnts(config_->NoOfPartitions() * size, 0);
  std::vector<int64_t> outPartitions;
  outPartitions.reserve(length);
  Status status = HashPartitionArray(cylon::ToArrowPool(ctx_.get()), arr,
                                     hash_targets, &outPartitions, cnts);
  std::string s = " ";
  for (int i = 0; i < length; i++) {
    outPartitions[i] = outPartitions[i] / size;
  }
  // now split
  for (int i = 0; i < table->num_columns(); i++) {
    std::shared_ptr<arrow::DataType> type = table->column(i)->chunk(0)->type();
    std::shared_ptr<arrow::Array> array = table->column(i)->chunk(0);

    std::shared_ptr<ArrowArrayStreamingSplitKernel> splitKernel = received_tables_[i];
    splitKernel->Split(array, outPartitions);
  }
  return true;
}

void cylon::SplitOp::OnParentsFinalized() {

}

bool cylon::SplitOp::Finalize() {
  // keep arrays for each target, these arrays are used for creating the table
  std::unordered_map<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>> data_arrays;
  int kI = config_->NoOfPartitions();
  for (int t = 0; t < kI; t++) {
    data_arrays.insert(
        std::pair<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>>(
            t, std::make_shared<std::vector<std::shared_ptr<arrow::Array>>>()));
  }
  
  // now we can finish the streaming kernels
  for (int i = 0; i < schema->num_fields(); i++) {
    std::shared_ptr<ArrowArrayStreamingSplitKernel> splitKernel = received_tables_[i];
    std::unordered_map<int, std::shared_ptr<arrow::Array>> splited_arrays;
    splitKernel->finish(splited_arrays);

    for (const auto &x : splited_arrays) {
      std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>> cols = data_arrays[x.first];
      cols->push_back(x.second);
    }
  }

  for (int i = 0; i < kI; i++) {
    std::shared_ptr<arrow::Table> t = arrow::Table::Make(schema, *data_arrays[i]);
    std::shared_ptr<cylon::Table> kY = std::make_shared<cylon::Table>(t, ctx_.get());
    this->InsertToAllChildren(id, kY);
  }
  return true;
}

SplitOpConfig::SplitOpConfig(int no_of_partitions, std::shared_ptr<std::vector<int>> hash_columns) {
  this->no_of_partitions = no_of_partitions;
  this->hash_columns = std::move(hash_columns);
  LOG(INFO) << "Done creating split config";
}

int SplitOpConfig::NoOfPartitions() {
  return no_of_partitions;
}

std::shared_ptr<std::vector<int>> SplitOpConfig::HashColumns() {
  return hash_columns;
}
std::shared_ptr<SplitOpConfig> SplitOpConfig::Make(int no_partitions,
    const std::vector<int> &hash_cols) {
  std::shared_ptr<std::vector<int>> v = std::make_shared<std::vector<int>>(
      hash_cols.begin(), hash_cols.end());
  return std::make_shared<SplitOpConfig>(no_partitions, v);
}

}


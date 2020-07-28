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

#include <utility>
#include <unordered_map>
#include <memory>
#include <chrono>

#include "arrow_join.hpp"
#include "../join/join.hpp"
#include "arrow_partition_kernels.hpp"
#include "arrow_kernels.hpp"

namespace cylon {

ArrowJoin::ArrowJoin(cylon::CylonContext *ctx,
                     const std::vector<int> &source,
                     const std::vector<int> &targets,
                     int leftEdgeId,
                     int rightEdgeId,
                     cylon::JoinCallback *callback,
                     std::shared_ptr<arrow::Schema> schema,
                     arrow::MemoryPool *pool) {
  joinCallBack_ = callback;
  workerId_ = ctx->GetRank();
  leftCallBack_ = std::make_shared<AllToAllCallback>(&leftTables_);
  rightCallBack_ = std::make_shared<AllToAllCallback>(&rightTables_);
  leftAllToAll_ =
      std::make_shared<ArrowAllToAll>(ctx, source, targets, leftEdgeId, leftCallBack_, schema,
                                      pool);
  rightAllToAll_ =
      std::make_shared<ArrowAllToAll>(ctx, source, targets, rightEdgeId, rightCallBack_, schema,
                                      pool);
}

bool ArrowJoin::isComplete() {
  bool left = leftAllToAll_->isComplete();
  bool right = rightAllToAll_->isComplete();

  if (left && right) {
    LOG(INFO) << "Received everything to join";
    auto start = std::chrono::high_resolution_clock::now();
    std::shared_ptr<arrow::Table> joined_table;
    arrow::Status status = join::joinTables(leftTables_, rightTables_,
                                            cylon::join::config::JoinConfig::InnerJoin(0, 0),
                                            &joined_table,
                                            arrow::default_memory_pool());
    if (status != arrow::Status::OK()) {
      LOG(FATAL) << "Failed to join - error: " << status.CodeAsString();
      return true;
    }
    joinCallBack_->onJoin(joined_table);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    LOG(INFO) << workerId_ << "Total join time " + std::to_string(duration.count());
    return true;
  }
  return false;
}

AllToAllCallback::AllToAllCallback(std::vector<std::shared_ptr<arrow::Table>> *table) {
  tables_ = table;
}

bool AllToAllCallback::onReceive(int source, std::shared_ptr<arrow::Table> table) {
  tables_->push_back(table);
  return true;
}

ArrowJoinWithPartition::ArrowJoinWithPartition(cylon::CylonContext *ctx,
                                               const std::vector<int> &source,
                                               const std::vector<int> &targets, int leftEdgeId,
                                               int rightEdgeId,
                                               JoinCallback *callback,
                                               std::shared_ptr<arrow::Schema> schema,
                                               arrow::MemoryPool *pool, int leftColumnIndex,
                                               int rightColumnIndex) {
  workerId_ = ctx->GetRank();
  finished_ = false;
  targets_ = targets;
  leftColumnIndex_ = leftColumnIndex;
  rightColumnIndex_ = rightColumnIndex;
  join_ = std::make_shared<ArrowJoin>(ctx, source, targets, leftEdgeId, rightEdgeId, callback,
                                      schema, pool);
}

bool ArrowJoinWithPartition::isComplete() {
  if (!leftUnPartitionedTables_.empty()) {
    std::shared_ptr<arrow::Table> left_tab = leftUnPartitionedTables_.front();
    // keep arrays for each target, these arrays are used for creating the table
    std::unordered_map<int,
                      std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>> data_arrays;

    for (int t : targets_) {
      data_arrays.insert(
          std::pair<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>>(t,
             std::make_shared<std::vector<std::shared_ptr<arrow::Array>>>()));
    }

    auto column = left_tab->column(leftColumnIndex_);
    std::vector<int64_t> outPartitions;
    std::shared_ptr<arrow::Array> array = column->chunk(0);
    // first we partition the table
    cylon::Status status = HashPartitionArray(pool_, array, targets_, &outPartitions);
    if (!status.is_ok()) {
      LOG(FATAL) << "Failed to create the hash partition";
      return true;
    }

    for (int i = 0; i < left_tab->num_columns(); i++) {
      std::shared_ptr<arrow::DataType> type = array->type();
      std::shared_ptr<ArrowArraySplitKernel> splitKernel;
      status = CreateSplitter(type, pool_, &splitKernel);
      if (!status.is_ok()) {
        LOG(FATAL) << "Failed to create the splitter";
        return true;
      }

      // this one outputs arrays for each target as a map
      std::unordered_map<int, std::shared_ptr<arrow::Array>> arrays;
      splitKernel->Split(array, outPartitions, targets_, arrays);

      for (const auto &x : arrays) {
        std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>> cols = data_arrays[x.first];
        cols->push_back(x.second);
      }
    }
    // now insert these array to
    for (const auto &x : data_arrays) {
      std::shared_ptr<arrow::Table> table = arrow::Table::Make(left_tab->schema(), *x.second);
      join_->leftInsert(table, x.first);
    }
    leftUnPartitionedTables_.pop();
  }

  if (!rightUnPartitionedTables_.empty()) {
    std::shared_ptr<arrow::Table> left_tab = rightUnPartitionedTables_.front();
    // keep arrays for each target, these arrays are used for creating the table
    std::unordered_map<int,
                       std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>> data_arrays;
    for (int t : targets_) {
      data_arrays.insert(
          std::pair<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>>(
              t, std::make_shared<std::vector<std::shared_ptr<arrow::Array>>>()));
    }

    auto column = left_tab->column(rightColumnIndex_);
    std::vector<int64_t> outPartitions;
    std::shared_ptr<arrow::Array> array = column->chunk(0);
    // first we partition the table
    cylon::Status status = HashPartitionArray(pool_, array, targets_, &outPartitions);
    if (!status.is_ok()) {
      LOG(FATAL) << "Failed to create the hash partition";
      return true;
    }

    for (int i = 0; i < left_tab->num_columns(); i++) {
      std::shared_ptr<arrow::DataType> type = array->type();
      std::shared_ptr<ArrowArraySplitKernel> splitKernel;
      status = CreateSplitter(type, pool_, &splitKernel);
      if (!status.is_ok()) {
        LOG(FATAL) << "Failed to create the splitter";
        return true;
      }

      // this one outputs arrays for each target as a map
      std::unordered_map<int, std::shared_ptr<arrow::Array>> arrays;
      splitKernel->Split(array, outPartitions, targets_, arrays);

      for (const auto &x : arrays) {
        std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>> cols = data_arrays[x.first];
        cols->push_back(x.second);
      }
    }
    // now insert these array to
    for (const auto &x : data_arrays) {
      std::shared_ptr<arrow::Table> table = arrow::Table::Make(left_tab->schema(), *x.second);
      join_->rightInsert(table, x.first);
    }
    rightUnPartitionedTables_.pop();
  }
  return finished_ && rightUnPartitionedTables_.empty() && leftUnPartitionedTables_.empty() &&
         join_->isComplete();
}

}  // namespace cylon




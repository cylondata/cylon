#include "arrow_join.hpp"
#include "../join/join.hpp"
#include "arrow_partition_kernels.hpp"
#include "arrow_kernels.hpp"
#include <chrono>

namespace twisterx {
ArrowJoin::ArrowJoin(int worker_id, const std::vector<int> &source, const std::vector<int> &targets, int leftEdgeId,
                     int rightEdgeId, twisterx::JoinCallback *callback, std::shared_ptr<arrow::Schema> schema,
                     arrow::MemoryPool *pool) {
  joinCallBack_ = callback;
  workerId_ = worker_id;
  leftCallBack_ = std::make_shared<AllToAllCallback>(&leftTables_);
  rightCallBack_ = std::make_shared<AllToAllCallback>(&rightTables_);
  leftAllToAll_ =
      std::make_shared<ArrowAllToAll>(worker_id, source, targets, leftEdgeId, leftCallBack_, schema, pool);
  rightAllToAll_ =
      std::make_shared<ArrowAllToAll>(worker_id, source, targets, rightEdgeId, rightCallBack_, schema, pool);
}

bool ArrowJoin::isComplete() {
  bool left = leftAllToAll_->isComplete();
  bool right = rightAllToAll_->isComplete();

  if (left && right) {
    LOG(INFO) << "Received everything to join";
    auto start = std::chrono::high_resolution_clock::now();
    std::shared_ptr<arrow::Table> joined_table;
    arrow::Status status = join::joinTables(leftTables_, rightTables_, (int64_t) 0, (int64_t) 0,
                                            twisterx::join::config::INNER,
                                            twisterx::join::config::SORT,
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

ArrowJoinWithPartition::ArrowJoinWithPartition(int worker_id, const std::vector<int> &source,
                                               const std::vector<int> &targets, int leftEdgeId, int rightEdgeId,
                                               JoinCallback *callback, std::shared_ptr<arrow::Schema> schema,
                                               arrow::MemoryPool *pool, int leftColumnIndex, int rightColumnIndex) {
  workerId_ = worker_id;
  finished_ = false;
  targets_ = targets;
  leftColumnIndex_ = leftColumnIndex;
  rightColumnIndex_ = rightColumnIndex;
  join_ = std::make_shared<ArrowJoin>(worker_id, source, targets, leftEdgeId, rightEdgeId, callback, schema, pool);
}

bool ArrowJoinWithPartition::isComplete() {
  if (!leftUnPartitionedTables_.empty()) {
    std::shared_ptr<arrow::Table> left_tab = leftUnPartitionedTables_.front();
    // keep arrays for each target, these arrays are used for creating the table
    std::unordered_map<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>> data_arrays;

    for (int t : targets_) {
      data_arrays.insert(std::pair<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>>(t,
                                                                                                     std::make_shared<
                                                                                                         std::vector<std::shared_ptr<
                                                                                                             arrow::Array>>>()));
    }

    auto column = left_tab->column(leftColumnIndex_);
    std::vector<int64_t> outPartitions;
    std::shared_ptr<arrow::Array> array = column->chunk(0);
    // first we partition the table
    arrow::Status status = HashPartitionArray(pool_, array, targets_, &outPartitions);
    if (status != arrow::Status::OK()) {
      LOG(FATAL) << "Failed to create the hash partition";
      return true;
    }

    for (int i = 0; i < left_tab->num_columns(); i++) {
      std::shared_ptr<arrow::DataType> type = array->type();
      std::unique_ptr<ArrowArraySplitKernel> splitKernel;
      status = CreateSplitter(type, pool_, &splitKernel);
      if (status != arrow::Status::OK()) {
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
    std::unordered_map<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>> data_arrays;
    for (int t : targets_) {
      data_arrays.insert(
          std::pair<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>>(
              t, std::make_shared<std::vector<std::shared_ptr<arrow::Array>>>()));
    }

    auto column = left_tab->column(rightColumnIndex_);
    std::vector<int64_t> outPartitions;
    std::shared_ptr<arrow::Array> array = column->chunk(0);
    // first we partition the table
    arrow::Status status = HashPartitionArray(pool_, array, targets_, &outPartitions);
    if (status != arrow::Status::OK()) {
      LOG(FATAL) << "Failed to create the hash partition";
      return true;
    }

    for (int i = 0; i < left_tab->num_columns(); i++) {
      std::shared_ptr<arrow::DataType> type = array->type();
      std::unique_ptr<ArrowArraySplitKernel> splitKernel;
      status = CreateSplitter(type, pool_, &splitKernel);
      if (status != arrow::Status::OK()) {
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
  return finished_ && rightUnPartitionedTables_.empty() && leftUnPartitionedTables_.empty() && join_->isComplete();
}

}




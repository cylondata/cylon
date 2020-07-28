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

#ifndef CYLON_ARROW_JOIN_H
#define CYLON_ARROW_JOIN_H

#include <vector>
#include <glog/logging.h>

#include "arrow_all_to_all.hpp"

namespace cylon {
class JoinCallback {
 public:
  /**
     * This function is called when a data is received
     * @param source the source
     * @param buffer the buffer allocated by the system, we need to free this
     * @param length the length of the buffer
     * @return true if we accept this buffer
     */
  virtual bool onJoin(std::shared_ptr<arrow::Table> table) = 0;
};

class AllToAllCallback : public ArrowCallback {
 public:
  explicit AllToAllCallback(std::vector<std::shared_ptr<arrow::Table>> *table);
  /**
   * The receive callback with the arrow table
   * @param source source
   * @param table the table
   * @return true if the table is accepted
   */
  bool onReceive(int source, std::shared_ptr<arrow::Table> table) override;
 private:
  std::vector<std::shared_ptr<arrow::Table>> *tables_;
};

class ArrowJoin {
 public:
  /**
     * Constructor
     * @param worker_id
     * @param all_workers
     * @return
     */
  ArrowJoin(cylon::CylonContext *ctx,
            const std::vector<int> &source,
            const std::vector<int> &targets,
            int leftEdgeId,
            int rightEdgeId,
            JoinCallback *callback,
            std::shared_ptr<arrow::Schema> schema,
            arrow::MemoryPool *pool);

  /**
   * Insert a partitioned table, this table will be sent directly
   *
   * @param buffer the buffer to send
   * @param length the length of the message
   * @param target the target to send the message
   * @return true if the buffer is accepted
   */
  int leftInsert(const std::shared_ptr<arrow::Table> &table, int target) {
	return leftAllToAll_->insert(table, target);
  }

  /**
   * Insert a partitioned table, this table will be sent directly
   * @param table
   * @param target
   * @return
   */
  int rightInsert(const std::shared_ptr<arrow::Table> &table, int target) {
	return rightAllToAll_->insert(table, target);
  }

  /**
   * Check weather the operation is complete, this method needs to be called until the operation is complete
   * @return true if the operation is complete
   */
  bool isComplete();

  /**
   * When this function is called, the operation finishes at both receivers and targets
   * @return
   */
  void finish() {
	leftAllToAll_->finish();
	rightAllToAll_->finish();
  }

  /*
   * Close the operation
   */
  void close() {
	leftAllToAll_->close();
	rightAllToAll_->close();
  }

 private:
  std::shared_ptr<ArrowAllToAll> leftAllToAll_;
  std::shared_ptr<ArrowAllToAll> rightAllToAll_;
  std::vector<std::shared_ptr<arrow::Table>> leftTables_;
  std::vector<std::shared_ptr<arrow::Table>> rightTables_;
  std::shared_ptr<AllToAllCallback> leftCallBack_;
  std::shared_ptr<AllToAllCallback> rightCallBack_;
  cylon::JoinCallback *joinCallBack_;
  int workerId_;
};

class ArrowJoinWithPartition {
 public:
  /**
   * Constructor
   * @param worker_id
   * @param all_workers
   * @return
   */
  ArrowJoinWithPartition(cylon::CylonContext *ctx,
                         const std::vector<int> &source,
                         const std::vector<int> &targets,
                         int leftEdgeId,
                         int rightEdgeId,
                         JoinCallback *callback,
                         std::shared_ptr<arrow::Schema> schema,
                         arrow::MemoryPool *pool,
                         int leftColumnIndex,
                         int rightColumnIndex);

  /**
   * Insert a partitioned table, this table will be sent directly
   *
   * @param buffer the buffer to send
   * @param length the length of the message
   * @param target the target to send the message
   * @return true if the buffer is accepted
   */
  int leftInsert(const std::shared_ptr<arrow::Table> &table) {
	leftUnPartitionedTables_.push(table);
	return 1;
  }

  /**
   * Insert a partitioned table, this table will be sent directly
   * @param table
   * @param target
   * @return
   */
  int rightInsert(const std::shared_ptr<arrow::Table> &table) {
	rightUnPartitionedTables_.push(table);
    return 1;
  }

  /**
   * Check weather the operation is complete, this method needs to be called until the operation is complete
   * @return true if the operation is complete
   */
  bool isComplete();

  /**
   * When this function is called, the operation finishes at both receivers and targets
   * @return
   */
  void finish() {
	finished_ = true;
  }

  /*
   * Close the operation
   */
  void close() {
	join_->close();
  }
 private:
  // keep track of the un partitioned tables
  std::queue<std::shared_ptr<arrow::Table>> leftUnPartitionedTables_;
  std::queue<std::shared_ptr<arrow::Table>> rightUnPartitionedTables_;
  std::shared_ptr<ArrowJoin> join_;

  int workerId_;
  bool finished_;
  arrow::MemoryPool *pool_;
  std::vector<int32_t> targets_;
  int leftColumnIndex_;
  int rightColumnIndex_;
};

}

#endif //CYLON_ARROW_JOIN_H

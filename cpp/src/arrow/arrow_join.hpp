#ifndef TWISTERX_ARROW_JOIN_H
#define TWISTERX_ARROW_JOIN_H

#include <vector>
#include <glog/logging.h>

#include "arrow_all_to_all.hpp"

namespace twisterx {
  class JoinCallback {
  public:
    /**
       * This function is called when a data is received
       * @param source the source
       * @param buffer the buffer allocated by the system, we need to free this
       * @param length the length of the buffer
       * @return true if we accept this buffer
       */
    virtual bool onJoin(std::shared_ptr <arrow::Table> table) = 0;
  };

  class AllToAllCallback : public ArrowCallback {
  public:
    AllToAllCallback(std::vector<std::shared_ptr<arrow::Table>>* table);
    /**
     * The receive callback with the arrow table
     * @param source source
     * @param table the table
     * @return true if the table is accepted
     */
    bool onReceive(int source, std::shared_ptr <arrow::Table> table);
  private:
    std::vector<std::shared_ptr<arrow::Table>>* tables_;
  };


  class ArrowJoin {
  public:
    /**
       * Constructor
       * @param worker_id
       * @param all_workers
       * @return
       */
    ArrowJoin(int worker_id, const std::vector<int> &source, const std::vector<int> &targets, int leftEdgeId, int rightEdgeId,
              JoinCallback *callback, std::shared_ptr <arrow::Schema> schema);

    /**
     * Insert a buffer to be sent, if the buffer is accepted return true
     *
     * @param buffer the buffer to send
     * @param length the length of the message
     * @param target the target to send the message
     * @return true if the buffer is accepted
     */
    int leftInsert(const std::shared_ptr <arrow::Table> &table, int target) {
      return leftAllToAll_->insert(table, target);
    }

    int rightInsert(const std::shared_ptr <arrow::Table> &table, int target) {
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
    twisterx::JoinCallback *joinCallBack_;
  };
}


#endif //TWISTERX_ARROW_JOIN_H

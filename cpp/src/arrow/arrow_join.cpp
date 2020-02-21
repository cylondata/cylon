#include "arrow_join.hpp"
#include "../join/tx_join.hpp";

namespace twisterx {
  ArrowJoin::ArrowJoin(int worker_id, const std::vector<int> &source, const std::vector<int> &targets, int edgeId,
                       twisterx::JoinCallback *callback, std::shared_ptr<arrow::Schema> schema) {
    leftCallBack_ = std::make_shared<AllToAllCallback>(&leftTables_);
    rightCallBack_ = std::make_shared<AllToAllCallback>(&rightTables_);
    leftAllToAll_ = std::make_shared<ArrowAllToAll>(worker_id, source, targets, edgeId, leftCallBack_.get(), schema);
    rightAllToAll_ = std::make_shared<ArrowAllToAll>(worker_id, source, targets, edgeId, rightCallBack_.get(), schema);
  }

  bool ArrowJoin::isComplete() {
    bool left = leftAllToAll_->isComplete();
    bool right = leftAllToAll_->isComplete();

    if (left && right) {
      LOG(INFO) << "Received everything to join";
      // join
      return true;
    }
    return false;
  }

  AllToAllCallback::AllToAllCallback(std::vector<std::shared_ptr<arrow::Table>>* table) {
    tables_ = table;
  }

  bool AllToAllCallback::onReceive(int source, std::shared_ptr<arrow::Table> table) {
    tables_->push_back(table);
  }
}




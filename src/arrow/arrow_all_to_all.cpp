#include "arrow_all_to_all.hpp"

namespace twisterx {
  ArrowAllToAll::ArrowAllToAll(int worker_id, const std::vector<int> &source, const std::vector<int> &targets,
                               int edgeId, twisterx::ReceiveCallback *callback) {
    targets_ = targets;
    srcs_ = source;
    
    // we need to pass the correct arguments
    all_ = std::make_shared<AllToAll>(worker_id, source, targets, edgeId, callback);

    // add the trackers for sending
    for (auto t : targets) {
      inputs_.insert(std::pair<int, PendingSendTable>(t, PendingSendTable()));
    }
  }

  int ArrowAllToAll::insert(const std::shared_ptr<arrow::Table>& arrow, int target) {
    //todo: check weather we have enough memory

    // lets save the table into pending and move on
    PendingSendTable &st = inputs_[target];
    st.pending.push(arrow);
    return 1;
  }

  bool ArrowAllToAll::isComplete() {
    // we need to send the buffers
    for (auto t : inputs_) {
      if (t.second.status == ARROW_HEADER_INIT) {
        std::queue<std::shared_ptr<arrow::Table>> &pend = t.second.pending;
        if (!pend.empty()) {
          t.second.currentTable = t.second.pending.front();
          t.second.pending.pop();
        }
      } else if (t.second.status == ARROW_HEADER_COLUMN_CONTINUE) {
        const std::shared_ptr<arrow::ChunkedArray> &cArr = t.second.currentTable->column(t.second.columnIndex);
        const std::shared_ptr<arrow::Array> &arr = cArr->chunk(t.second.arrayIndex);
        const std::shared_ptr<arrow::ArrayData> &data = arr->data();
        std::shared_ptr<arrow::Buffer> &buf = data->buffers[t.second.bufferIndex];
        // lets send this buffer, we need to send the length at this point
        all_->insert((void *)buf->data(), (int) buf->size(), t.first);
      } else if (t.second.status == ARROW_HEADER_COLUMN_END) {

      }
    }
    return all_->isComplete();
  }

  void ArrowAllToAll::finish() {
    all_->finish();
  }

  void ArrowAllToAll::close() {
    // clear the input map
    inputs_.clear();
    // call close on the underlying allto all
    all_->close();
  }

  bool ArrowAllToAll::onReceive(int source, void *buffer, int length) {
    return false;
  }

  bool ArrowAllToAll::onReceiveHeader(int source, int *buffer, int length) {
    return false;
  }
}
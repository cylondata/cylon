#include "arrow_all_to_all.hpp"

namespace twisterx {
  ArrowAllToAll::ArrowAllToAll(int worker_id, const std::vector<int> &source, const std::vector<int> &targets,
                               int edgeId, ArrowCallback *callback) {
    targets_ = targets;
    srcs_ = source;
    recv_callback_ = callback;
    
    // we need to pass the correct arguments
    all_ = std::make_shared<AllToAll>(worker_id, source, targets, edgeId, this);

    // add the trackers for sending
    for (auto t : targets) {
      inputs_.insert(std::pair<int, PendingSendTable>(t, PendingSendTable()));
    }
    
    for (auto t : source) {
      receives_.insert(std::pair<int, PendingReceiveTable>(t, PendingReceiveTable()));
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
      }

      if (t.second.status == ARROW_HEADER_COLUMN_CONTINUE) {
        int noOfColumns = t.second.currentTable->columns().size();
        bool canContinue = true;
        while (t.second.columnIndex < noOfColumns && canContinue) {
          const std::shared_ptr<arrow::ChunkedArray> &cArr = t.second.currentTable->column(t.second.columnIndex);

          unsigned long size = cArr->chunks().size();
          while (t.second.arrayIndex < size && canContinue) {
            const std::shared_ptr<arrow::Array> &arr = cArr->chunk(t.second.arrayIndex);

            const std::shared_ptr<arrow::ArrayData> &data = arr->data();
            while (t.second.bufferIndex < data->buffers.size()) {
              std::shared_ptr<arrow::Buffer> &buf = data->buffers[t.second.bufferIndex];
              int hdr[4];
              hdr[0] = t.second.columnIndex;
              hdr[1] = t.second.arrayIndex;
              hdr[2] = t.second.bufferIndex;
              hdr[3] = data->buffers.size();
              // lets send this buffer, we need to send the length at this point
              bool accept = all_->insert((void *) buf->data(), (int) buf->size(), t.first, hdr, 4);
              if (!accept) {
                canContinue = false;
                break;
              }
              t.second.bufferIndex++;
            }
            // if we can continue, that means we are finished with this array
            if (canContinue) {
              t.second.arrayIndex++;
            }
          }
          // if we can continue, that means we are finished with this column
          if (canContinue) {
            t.second.columnIndex++;
          }
        }

        // if we are at this stage, we have sent everything for this , so lets resets
        if (canContinue) {
          t.second.columnIndex = 0;
          t.second.arrayIndex = 0;
          t.second.bufferIndex = 0;
          // we are done with this target, for this call
          t.second.status = ARROW_HEADER_INIT;
        }
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
    PendingReceiveTable &table = receives_[source];
    return false;
  }

  bool ArrowAllToAll::onReceiveHeader(int source, int *buffer, int length) {
    PendingReceiveTable &table = receives_[source];
    table.columnIndex = buffer[0];
    table.arrayIndex = buffer[1];
    table.bufferIndex = buffer[2];
    table.noBuffers = buffer[3];
    return true;
  }
}
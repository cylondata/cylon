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
#include <glog/logging.h>

#include <vector>
#include <utility>
#include <string>
#include <memory>

#include "arrow_all_to_all.hpp"


namespace cylon {
ArrowAllToAll::ArrowAllToAll(cylon::CylonContext *ctx,
                             const std::vector<int> &source,
                             const std::vector<int> &targets,
                             int edgeId,
                             std::shared_ptr<ArrowCallback> callback,
                             std::shared_ptr<arrow::Schema> schema,
                             arrow::MemoryPool *pool) {
  targets_ = targets;
  srcs_ = source;
  recv_callback_ = callback;
  schema_ = schema;
  receivedBuffers_ = 0;
  workerId_ = ctx->GetRank();
  pool_ = pool;
  completed_ = false;
  finishCalled_ = false;

  // we need to pass the correct arguments
  all_ = std::make_shared<AllToAll>(ctx, source, targets, edgeId, this);

  // add the trackers for sending
  for (auto t : targets) {
    inputs_.insert(std::pair<int, std::shared_ptr<PendingSendTable>>(t,
        std::make_shared<PendingSendTable>()));
  }

  for (auto t : source) {
    receives_.insert(std::pair<int, std::shared_ptr<PendingReceiveTable>>(t,
        std::make_shared<PendingReceiveTable>()));
  }
}

int ArrowAllToAll::insert(const std::shared_ptr<arrow::Table> &arrow, int target) {
  // todo: check weather we have enough memory
  // lets save the table into pending and move on
  std::shared_ptr<PendingSendTable> st = inputs_[target];
  st->pending.push(arrow);
  return 1;
}

bool ArrowAllToAll::isComplete() {
  if (completed_) {
    return true;
  }
  bool isAllEmpty = true;
  // we need to send the buffers
  for (auto t : inputs_) {
    if (t.second->status == ARROW_HEADER_INIT) {
      if (!t.second->pending.empty()) {
        t.second->currentTable = t.second->pending.front();
        t.second->pending.pop();
        t.second->status = ARROW_HEADER_COLUMN_CONTINUE;
      }
    }

    if (t.second->status == ARROW_HEADER_COLUMN_CONTINUE) {
      int noOfColumns = t.second->currentTable->columns().size();
      bool canContinue = true;
      while (t.second->columnIndex < noOfColumns && canContinue) {
        std::shared_ptr<arrow::ChunkedArray> cArr = t.second->currentTable->column(
            t.second->columnIndex);

        uint64_t size = cArr->chunks().size();
        while (static_cast<size_t>(t.second->arrayIndex) < size && canContinue) {
          std::shared_ptr<arrow::Array> arr = cArr->chunk(t.second->arrayIndex);

          std::shared_ptr<arrow::ArrayData> data = arr->data();
          while (static_cast<size_t>(t.second->bufferIndex) < data->buffers.size()) {
            std::shared_ptr<arrow::Buffer> buf = data->buffers[t.second->bufferIndex];
            int hdr[5];
            hdr[0] = t.second->columnIndex;
            hdr[1] = t.second->bufferIndex;
            hdr[2] = data->buffers.size();
            hdr[3] = cArr->chunks().size();
            hdr[4] = data->length;
            // lets send this buffer, we need to send the length at this point
            bool accept = all_->insert(buf->mutable_data(),
                static_cast<int>(buf->size()), t.first, hdr, 5);
            if (!accept) {
              canContinue = false;
              break;
            }
            t.second->bufferIndex++;
          }
          // if we can continue, that means we are finished with this array
          if (canContinue) {
            t.second->bufferIndex = 0;
            t.second->arrayIndex++;
          }
        }
        // if we can continue, that means we are finished with this column
        if (canContinue) {
          t.second->arrayIndex = 0;
          t.second->columnIndex++;
        }
      }

      // if we are at this stage, we have sent everything for this , so lets resets
      if (canContinue) {
        t.second->columnIndex = 0;
        t.second->arrayIndex = 0;
        t.second->bufferIndex = 0;
        // we are done with this target, for this call
        t.second->status = ARROW_HEADER_INIT;
      }
    }

    if (!t.second->pending.empty() || t.second->status == ARROW_HEADER_COLUMN_CONTINUE) {
      isAllEmpty = false;
    }
  }

  if (isAllEmpty && finished && !finishCalled_) {
    all_->finish();
    finishCalled_ = true;
  }

  // if completed gets true, we will never reach this point
  completed_ = isAllEmpty && all_->isComplete() && finishedSources_.size() == srcs_.size();
  return completed_;
}

void ArrowAllToAll::finish() {
  finished = true;
}

void ArrowAllToAll::close() {
  // clear the input map
  inputs_.clear();
  // call close on the underlying allto all
  all_->close();
}

void debug(int thisWorker, std::string msg) {
  if (thisWorker == -1) {
    LOG(INFO) << msg;
  }
}

bool ArrowAllToAll::onReceive(int source, void *buffer, int length) {
  std::shared_ptr<PendingReceiveTable> table = receives_[source];
  receivedBuffers_++;
  // create the buffer hosting the value
  std::shared_ptr<arrow::Buffer> buf = std::make_shared<arrow::Buffer>(
      static_cast<uint8_t *>(buffer), length);
  table->buffers.push_back(buf);
  // now check weather we have the expected number of buffers received
  if (table->noBuffers == table->bufferIndex + 1) {
    // okay we are done with this array
    std::shared_ptr<arrow::ArrayData> data = arrow::ArrayData::Make(
        schema_->field(table->columnIndex)->type(), table->length, table->buffers);
    // clears the buffers
    table->buffers.clear();
    // create an array
    std::shared_ptr<arrow::Array> array = arrow::MakeArray(data);
    table->arrays.push_back(array);

    // we have received all the arrays of the chunk array
    if (table->arrays.size() == static_cast<size_t>(table->noArray)) {
      std::shared_ptr<arrow::ChunkedArray> chunkedArray = std::make_shared<arrow::ChunkedArray>(
          table->arrays, schema_->field(table->columnIndex)->type());
      // clear the arrays
      table->arrays.clear();
      table->currentArrays.push_back(chunkedArray);
      if (table->currentArrays.size() == static_cast<size_t>(schema_->num_fields())) {
        // now we can create the table
        std::shared_ptr<arrow::Table> tablePtr = arrow::Table::Make(schema_, table->currentArrays);
        // clear the current array
        table->currentArrays.clear();
        recv_callback_->onReceive(source, tablePtr);
      }
    }
  }

  return true;
}

bool ArrowAllToAll::onReceiveHeader(int source, int fin, int *buffer, int length) {
  if (!fin) {
    if (length != 5) {
      LOG(FATAL) << "Incorrect length on header, expected 5 ints got " << length;
      return false;
    }

    std::shared_ptr<PendingReceiveTable> table = receives_[source];
    table->columnIndex = buffer[0];
    table->bufferIndex = buffer[1];
    table->noBuffers = buffer[2];
    table->noArray = buffer[3];
    table->length = buffer[4];
  } else {
    finishedSources_.push_back(source);
  }
  return true;
}

bool ArrowAllToAll::onSendComplete(int target, void *buffer, int length) {
//    pool_->Free((uint8_t *)buffer, length);
  return false;
}

}  // namespace cylon

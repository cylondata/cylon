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

#include "all_to_all_op.hpp"

cylon::AllToAllOp::AllToAllOp(const std::shared_ptr<CylonContext> &ctx,
                              const std::shared_ptr<arrow::Schema> &schema,
                              int id,
                              const std::shared_ptr<ResultsCallback> &callback,
                              const std::shared_ptr<AllToAllOpConfig> &config)
    : Op(ctx, schema, id, callback) {
  class AllToAllListener : public cylon::ArrowCallback {
    AllToAllOp *shuffle_op;
    std::shared_ptr<CylonContext> ctx;

   public:
    explicit AllToAllListener(const std::shared_ptr<CylonContext> &ctx, AllToAllOp *shuffle_op) {
      this->shuffle_op = shuffle_op;
      this->ctx = ctx;
    }

    bool onReceive(int source, const std::shared_ptr<arrow::Table> &table, int tag) override {
      // todo check whether the const cast is appropriate
      auto tab = std::make_shared<cylon::Table>(
          const_cast<std::shared_ptr<arrow::Table> &>(table), ctx);
      this->shuffle_op->InsertToAllChildren(tag, tab);
      return true;
    };
  };
  this->finish_called_ = false;
  std::shared_ptr<AllToAllListener> all_to_all_listener = std::make_shared<AllToAllListener>(ctx, this);
  this->all_to_all_ = new cylon::ArrowAllToAll(const_cast<std::shared_ptr<CylonContext> &>(ctx),
                                               ctx->GetNeighbours(true),
                                               ctx->GetNeighbours(true), id,
                                               all_to_all_listener, schema);
}

bool cylon::AllToAllOp::Execute(int tag, std::shared_ptr<Table> &table) {
  if (!started_time) {
    start = std::chrono::high_resolution_clock::now();
    started_time = true;
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  // if the table is for the same worker pass to childern
  if (tag == ctx_->GetRank()) {
    this->InsertToAllChildren(this->GetId(), table);
  } else {
    // todo change here to use the tag appropriately
    this->all_to_all_->insert(table->get_table(), tag, this->GetId());
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  exec_time += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  return true;
}

bool cylon::AllToAllOp::IsComplete() {
  //LOG(INFO) << "Calling shuffle progress";
  this->all_to_all_->isComplete();
  return Op::IsComplete();
}

bool cylon::AllToAllOp::Finalize() {
  if (!finish_called_) {
    this->all_to_all_->finish();
    finish_called_ = true;
  }
  if (this->all_to_all_->isComplete()) {
    this->all_to_all_->close();
    auto t2 = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Shuffle time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - start).count();
    return true;
  }
  return false;
}

void cylon::AllToAllOp::OnParentsFinalized() {
}


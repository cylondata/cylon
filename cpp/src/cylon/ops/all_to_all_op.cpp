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

cylon::ShuffleOp::ShuffleOp(std::shared_ptr<cylon::CylonContext> ctx,
                            std::shared_ptr<arrow::Schema> schema,
                            int id,
                            shared_ptr<ResultsCallback> callback,
                            shared_ptr<ShuffleOpConfig> config) : Op(ctx, schema, id, callback) {
  class AllToAllListener : public cylon::ArrowCallback {
    ShuffleOp *shuffle_op;
    cylon::CylonContext *ctx;

   public:
    explicit AllToAllListener(cylon::CylonContext *ctx, ShuffleOp *shuffle_op) {
      this->shuffle_op = shuffle_op;
      this->ctx = ctx;
    }

    bool onReceive(int source, const std::shared_ptr<arrow::Table> &table, int reference) override {
      this->shuffle_op->InsertToAllChildren(reference, std::make_shared<cylon::Table>(table, ctx));
      return true;
    };
  };

  this->all_to_all_ = new cylon::ArrowAllToAll(&*ctx, ctx->GetNeighbours(true),
                                               ctx->GetNeighbours(true), id,
                                               std::make_shared<AllToAllListener>(&*ctx, this), schema);
}

bool cylon::ShuffleOp::Execute(int tag, shared_ptr<Table> table) {
  LOG(INFO) << "Executing shuffle";
  // if the table is for the same worker pass to childern
  if (tag == ctx_->GetRank()) {
    this->InsertToAllChildren(tag, table);
  } else {
    this->all_to_all_->insert(table->get_table(), tag);
  }
  return true;
}

void cylon::ShuffleOp::Progress() {
  LOG(INFO) << "Calling shuffle progress";
  this->all_to_all_->isComplete();
  Op::Progress();
}

bool cylon::ShuffleOp::Finalize() {
  LOG(INFO) << "Finalizing shuffle";
  if (this->all_to_all_->isComplete()) {
    this->all_to_all_->close();
    return true;
  }
  return false;
}

void cylon::ShuffleOp::OnParentsFinalized() {
  this->all_to_all_->finish();
}


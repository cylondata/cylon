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

cylon::AllToAllOp::AllToAllOp(const std::shared_ptr<cylon::CylonContext> &ctx,
                              const std::shared_ptr<arrow::Schema> &schema,
                              int id,
                              const std::shared_ptr<ResultsCallback> &callback,
                              const std::shared_ptr<AllToAllOpConfig> &config) :
                                Op(ctx, schema, id, callback) {
  class AllToAllListener : public cylon::ArrowCallback {
    AllToAllOp *shuffle_op;
    cylon::CylonContext *ctx;

   public:
    explicit AllToAllListener(cylon::CylonContext *ctx, AllToAllOp *shuffle_op) {
      this->shuffle_op = shuffle_op;
      this->ctx = ctx;
    }

    bool onReceive(int source, const std::shared_ptr<arrow::Table> &table, int tag) override {
      // todo check whether the const cast is appropriate
      LOG(INFO) << "received a table with tag" << tag;
      auto tab = std::make_shared<cylon::Table>(
          const_cast<std::shared_ptr<arrow::Table> &>(table), ctx);
      this->shuffle_op->InsertToAllChildren(tag, tab);
      return true;
    };
  };

  this->all_to_all_ = new cylon::ArrowAllToAll(ctx.get(), ctx->GetNeighbours(true),
                                           ctx->GetNeighbours(true), id,
                                           std::make_shared<AllToAllListener>(&*ctx, this), schema);
}

bool cylon::AllToAllOp::Execute(int tag, shared_ptr<Table> table) {
  LOG(INFO) << "Executing shuffle";
  // if the table is for the same worker pass to childern
  if (tag == ctx_->GetRank()) {
    LOG(INFO) << "Locally Sending a table with tag " << tag;
    this->InsertToAllChildren(this->GetId(), table);
  } else {
    LOG(INFO) << "Sending a table with tag " << tag;
    // todo change here to use the tag appropriately
    this->all_to_all_->insert(table->get_table(), tag);
  }
  return true;
}

bool cylon::AllToAllOp::IsComplete() {
  //LOG(INFO) << "Calling shuffle progress";
  this->all_to_all_->isComplete();
  return Op::IsComplete();
}

bool cylon::AllToAllOp::Finalize() {
  LOG(INFO) << "ying to finalize shuffle";
  if (this->all_to_all_->isComplete()) {
    this->all_to_all_->close();
    return true;
  }
  return false;
}

void cylon::AllToAllOp::OnParentsFinalized() {
  this->all_to_all_->finish();
}


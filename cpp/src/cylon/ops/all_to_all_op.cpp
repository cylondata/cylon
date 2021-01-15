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
#include "table.hpp"

cylon::AllToAllOp::AllToAllOp(const std::shared_ptr<CylonContext> &ctx,
                              const std::shared_ptr<arrow::Schema> &schema,
                              int id,
                              const ResultsCallback &callback,
                              const AllToAllOpConfig &config)
    : Op(ctx, schema, id, callback) {
  std::shared_ptr<CylonContext> ctx_cp = ctx;
  ArrowCallback all_to_all_listener = [&](int source, const std::shared_ptr<arrow::Table> &table, int tag) {
    auto tab = std::make_shared<cylon::Table>(const_cast<std::shared_ptr<arrow::Table> &>(table), ctx_cp);
    this->InsertToAllChildren(tag, tab);
    return true;
  };

  this->finish_called_ = false;
  this->all_to_all_ = new cylon::ArrowAllToAll(ctx_cp,
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


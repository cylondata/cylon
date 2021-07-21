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
#include <cylon/ops/join_op.hpp>

#include <chrono>

cylon::JoinOp::JoinOp(const std::shared_ptr<CylonContext> &ctx,
                      const std::shared_ptr<arrow::Schema> &schema,
                      int32_t id,
                      const ResultsCallback &callback,
                      const cylon::join::config::JoinConfig &config)
    : Op(ctx, schema, id, callback) {
  // initialize join kernel
  join_kernel_ = new cylon::kernel::JoinKernel(ctx, schema, &config);
}

bool cylon::JoinOp::Execute(int tag, std::shared_ptr<Table> &table) {
  // do join
  join_kernel_->InsertTable(tag, table);
  return true;
}

void cylon::JoinOp::OnParentsFinalized() {
  // do nothing
}

bool cylon::JoinOp::Finalize() {
  auto t1 = std::chrono::high_resolution_clock::now();
  // return finalize join
  std::shared_ptr<cylon::Table> final_result;
  this->join_kernel_->Finalize(final_result);
  this->InsertToAllChildren(0, final_result);
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Join time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  return true;
}

cylon::JoinOp::~JoinOp() {
  delete join_kernel_;
}

//int32_t cylon::JoinOpConfig::GetJoinColumn() const {
//  return join_column;
//}

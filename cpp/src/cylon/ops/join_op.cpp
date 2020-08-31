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

#include "join_op.hpp"

#include <utility>
#include "partition_op.hpp"

cylon::JoinOp::JoinOp(std::shared_ptr<CylonContext> ctx,
                      std::shared_ptr<arrow::Schema> schema,
                      int32_t  id,
                      std::shared_ptr<ResultsCallback> callback,
                      std::shared_ptr<JoinOpConfig> config) : Op(ctx, schema, id, callback) {
  this->config = config;
  // initialize join kernel
}

bool cylon::JoinOp::Execute(int tag, std::shared_ptr<Table> table) {
  // do join

  return true;
}

void cylon::JoinOp::OnParentsFinalized() {
  // do nothing
}

bool cylon::JoinOp::Finalize() {
  // return finalize join
  return false;
}

int32_t cylon::JoinOpConfig::GetJoinColumn() const {
  return join_column;
}

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

#include "union_op.hpp"

namespace cylon {

UnionOp::UnionOp(const std::shared_ptr<cylon::CylonContext> &ctx,
                 const std::shared_ptr<arrow::Schema> &schema,
                 int id,
                 const std::shared_ptr<ResultsCallback> &callback,
                 const std::shared_ptr<UnionOpConfig> &config) :
                 Op(ctx, schema, id, callback), config(config) {
  this->union_kernel = new cylon::kernel::Union(ctx, schema, config->GetExpectedRows());
}

bool UnionOp::Execute(int tag, std::shared_ptr<Table> table) {
  LOG(INFO) << "Executing local union";
  this->union_kernel->InsertTable(table);
  return true;
}

bool UnionOp::Finalize() {
  std::shared_ptr<cylon::Table> final_result;
  this->union_kernel->Finalize(final_result);
  this->InsertToAllChildren(0, final_result);
  return true;
}

void UnionOp::OnParentsFinalized() {
  // do nothing
}

int64_t UnionOpConfig::GetExpectedRows() const {
  return expected_rows;
}

void UnionOpConfig::SetExpectedRows(int64_t expected_final_rows) {
  UnionOpConfig::expected_rows = expected_final_rows;
}
}
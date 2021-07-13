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

#include "set_op.hpp"
#include "kernels/set_kernel.hpp"

namespace cylon {

UnionOp::UnionOp(const std::shared_ptr<CylonContext> &ctx,
                 const std::shared_ptr<arrow::Schema> &schema,
                 int id,
                 const ResultsCallback &callback,
                 const SetOpConfig &config,
                 cylon::kernel::SetOpType type)
    : Op(ctx, schema, id, callback) {
  union_kernel = cylon::kernel::CreateSetOp(ctx, schema, 0, type);
}

bool UnionOp::Execute(int tag, std::shared_ptr<Table> &table) {
  union_kernel->InsertTable(tag, table);
  return true;
}

bool UnionOp::Finalize() {
  std::shared_ptr<cylon::Table> final_result;
  union_kernel->Finalize(final_result);
  InsertToAllChildren(0, final_result);
  return true;
}

void UnionOp::OnParentsFinalized() {
  // do nothing
}

UnionOp::~UnionOp() {
}
}
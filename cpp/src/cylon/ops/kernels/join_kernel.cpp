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

#include <cylon/table.hpp>
#include <cylon/ctx/arrow_memory_pool_utils.hpp>
#include <cylon/ops/kernels/join_kernel.hpp>
#include "cylon/util/macros.hpp"

namespace cylon {
namespace kernel {

JoinKernel::JoinKernel(const std::shared_ptr<cylon::CylonContext> &ctx,
                       const std::shared_ptr<arrow::Schema> &schema,
                       const cylon::join::config::JoinConfig *join_config)
    : ctx(ctx), schema(schema), join_config(join_config) {}

void JoinKernel::InsertTable(int tag, const std::shared_ptr<cylon::Table> &table) {
  if (tag == 100) {
    left_tables.push(table->get_table());
  } else if (tag == 200) {
    right_tables.push(table->get_table());
  } else {
    LOG(FATAL) << "Un-recognized tag " << tag;
  }
}

cylon::Status JoinKernel::Finalize(std::shared_ptr<cylon::Table> &result) {
  size_t kI = left_tables.size();
  std::vector<std::shared_ptr<arrow::Table>> joined_tables;
  arrow::MemoryPool *kPool = cylon::ToArrowPool(this->ctx);
  for (size_t i = 0; i < kI; i++) {
    std::shared_ptr<arrow::Table> left_tab = left_tables.front();
    std::shared_ptr<arrow::Table> right_tab = right_tables.front();
    std::shared_ptr<arrow::Table> out;
    RETURN_CYLON_STATUS_IF_FAILED(join::JoinTables(left_tab, right_tab, *join_config, &out, kPool));
    left_tables.pop();
    right_tables.pop();
    joined_tables.push_back(out);
  }
  CYLON_ASSIGN_OR_RAISE(auto concat_tables,
                        arrow::ConcatenateTables(joined_tables,
                                                 arrow::ConcatenateTablesOptions::Defaults(),
                                                 kPool));

  return Table::FromArrowTable(ctx, std::move(concat_tables), result);
}

}  // namespace kernel
}  // namespace cylon

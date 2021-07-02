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

#include <ops/kernels/prepare_array.hpp>
#include <ctx/arrow_memory_pool_utils.hpp>
#include "union.hpp"

void cylon::kernel::Union::InsertTable(int tag, const std::shared_ptr<cylon::Table> &table) {
  if (tag == 100) {
    left_tables.push(table);
  } else if (tag == 200) {
    right_tables.push(table);
  } else {
    LOG(FATAL) << "Un-recognized tag " << tag;
  }
}

cylon::Status cylon::kernel::Union::Finalize(std::shared_ptr<cylon::Table> &result) {
  size_t kI = left_tables.size();
  arrow::MemoryPool *kPool = cylon::ToArrowPool(this->ctx);
  std::vector<std::shared_ptr<arrow::Table>> unioned_tables;
  for (size_t i = 0; i < kI; i++) {
    std::shared_ptr<cylon::Table> left_tab = left_tables.front();
    std::shared_ptr<cylon::Table> right_tab = right_tables.front();
    std::shared_ptr<cylon::Table> out;
    Status st = cylon::Union(left_tab, right_tab, out);
    left_tables.pop();
    right_tables.pop();
    unioned_tables.push_back(out->get_table());
  }
  arrow::Result<std::shared_ptr<arrow::Table>> concat_tables =
      arrow::ConcatenateTables(unioned_tables, arrow::ConcatenateTablesOptions::Defaults(), kPool);

  if (concat_tables.ok()) {
    auto final_table = concat_tables.ValueOrDie();
    LOG(INFO) << "Done concatenating tables, rows :  " << final_table->num_rows();
    cylon::Table::FromArrowTable(this->ctx, final_table, result);
    return Status(static_cast<int>(concat_tables.status().code()),
                  concat_tables.status().message());
  } else {
    return Status(static_cast<int>(concat_tables.status().code()),
                  concat_tables.status().message());
  }
}

cylon::kernel::Union::Union(const std::shared_ptr<CylonContext> &ctx,
                            const std::shared_ptr<arrow::Schema> &schema,
                            int64_t expected_rows) :
    schema(schema), ctx(ctx) {
}

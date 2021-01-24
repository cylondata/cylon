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

#include <ctx/arrow_memory_pool_utils.hpp>
#include "merge_op.hpp"

cylon::MergeOp::MergeOp(const std::shared_ptr<CylonContext> &ctx,
                        const std::shared_ptr<arrow::Schema> &schema,
                        int32_t id,
                        const std::shared_ptr<ResultsCallback> &callback) :
    Op(ctx, schema, id, callback) {
}

bool cylon::MergeOp::Execute(int tag, std::shared_ptr<Table> &table) {
  // do join
  if (received_tables_.find(tag) == received_tables_.end()) {
    received_tables_.insert(std::make_pair(tag, std::vector<std::shared_ptr<arrow::Table>>()));
  }
  received_tables_.find(tag)->second.push_back(table->get_table());
  return true;
}

void cylon::MergeOp::OnParentsFinalized() {
  // do nothing
}

bool cylon::MergeOp::Finalize() {
  // return finalize join
  // now we have the final set of tables
  std::vector<int32_t> to_remove;

  for (auto &received_table : received_tables_) {
    arrow::Result<std::shared_ptr<arrow::Table>> concat_tables =
        arrow::ConcatenateTables(received_table.second);

    if (concat_tables.ok()) {
      auto final_table = concat_tables.ValueOrDie();
      LOG(INFO) << "Done concatenating tables, rows :  " << final_table->num_rows();

      arrow::Result<std::shared_ptr<arrow::Table>> result = final_table->CombineChunks(cylon::ToArrowPool(this->ctx_));
      if (result.ok()) { // todo nothing can be done when this fails!
        std::shared_ptr<cylon::Table> t;
        cylon::Table::FromArrowTable(this->ctx_, result.ValueOrDie(), t);
        this->InsertToAllChildren(received_table.first, t);
      }

      // remove this group
      to_remove.push_back(received_table.first);
    } else {
      LOG(FATAL) << "Failed to concat the tables";
    }
  }

  for(auto key:to_remove){
    this->received_tables_.erase(key);
  }

  return received_tables_.empty();
}



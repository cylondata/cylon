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

cylon::MergeOp::MergeOp(std::shared_ptr<CylonContext> ctx,
                      std::shared_ptr<arrow::Schema> schema,
                      int32_t id,
                      std::shared_ptr<ResultsCallback> callback) :
    Op(std::move(ctx), std::move(schema), id,
       std::move(callback)) {
}

bool cylon::MergeOp::Execute(int tag, std::shared_ptr<Table> table) {
  LOG(INFO) << "Merge op";
  // do join
  received_tables_.push_back(table->get_table());
  return true;
}

void cylon::MergeOp::OnParentsFinalized() {
  // do nothing
}

bool cylon::MergeOp::Finalize() {
  // return finalize join
  // now we have the final set of tables
  LOG(INFO) << "Concatenating tables, Num of tables :  " << received_tables_.size();
  arrow::Result<std::shared_ptr<arrow::Table>> concat_tables =
      arrow::ConcatenateTables(received_tables_);

  if (concat_tables.ok()) {
    auto final_table = concat_tables.ValueOrDie();
    LOG(INFO) << "Done concatenating tables, rows :  " << final_table->num_rows();
    std::shared_ptr<arrow::Table> table_out;
    auto status = final_table->CombineChunks(cylon::ToArrowPool(this->ctx_.get()), &table_out);
    std::shared_ptr<cylon::Table> t;
    cylon::Table::FromArrowTable(this->ctx_.get(), table_out, &t);
    this->InsertToAllChildren(0, t);
  } else {
    LOG(FATAL) << "Failed to concat the tables";
  }
  return true;
}



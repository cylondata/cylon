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
#include "union.hpp"

void cylon::kernel::Union::InsertTable(std::shared_ptr<cylon::Table> table) {
  std::shared_ptr<arrow::Table> arrow_table;
  table->ToArrowTable(arrow_table);

  this->tables.push_back(arrow_table);
  this->indices_from_tabs.push_back(std::make_shared<std::vector<int64_t>>());

  auto const table_index = this->tables.size() - 1;
  for (int64_t i = 0; i < arrow_table->num_rows(); i++) {
    this->rows_set->insert(std::pair<int8_t, int64_t>(table_index, i));
  }
}

cylon::Status cylon::kernel::Union::Finalize(std::shared_ptr<cylon::Table> &result) {
  for (auto const &pr : *this->rows_set) {
    this->indices_from_tabs[pr.first]->push_back(pr.second);
  }

  std::vector<std::shared_ptr<arrow::ChunkedArray>> final_data_arrays;
  // prepare final arrays
  for (int32_t col_idx = 0; col_idx < this->schema->num_fields(); col_idx++) {
    arrow::ArrayVector array_vector;
    for (size_t tab_idx = 0; tab_idx < this->tables.size(); tab_idx++) {
      auto status = cylon::kernel::PrepareArray(this->ctx,
                                                tables[tab_idx],
                                                col_idx,
                                                indices_from_tabs[tab_idx],
                                                array_vector);

      if (!status.is_ok()) return status;
    }
    final_data_arrays.push_back(std::make_shared<arrow::ChunkedArray>(array_vector));
  }
  // create final table
  std::shared_ptr<arrow::Table> table = arrow::Table::Make(this->schema, final_data_arrays);
  auto merge_status = table->CombineChunks(cylon::ToArrowPool(&*this->ctx), &table);
  if (!merge_status.ok()) {
    return Status(static_cast<int>(merge_status.code()), merge_status.message());
  }
  result = std::make_shared<cylon::Table>(table, &*this->ctx);
  return cylon::Status::OK();
}

cylon::kernel::Union::Union(std::shared_ptr<cylon::CylonContext> ctx,
                            std::shared_ptr<arrow::Schema> schema,
                            int64_t expected_rows) {
  this->ctx = ctx;
  auto row_comparator = row_comparator(ctx,
                                       std::shared_ptr<std::vector<std::shared_ptr<arrow::Table>>>(&this->tables),
                                       schema);
  this->rows_set = new std::unordered_set<std::pair<int8_t, int64_t>, row_comparator, row_comparator>
      (expected_rows, row_comparator, row_comparator);
  this->schema = schema;
}

cylon::kernel::Union::~Union() {
  //delete this->rows_set;
}

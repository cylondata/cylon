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
#include "row_comparator.hpp"

void cylon::kernel::Union::InsertTable(const std::shared_ptr<cylon::Table> &table) {
  std::shared_ptr<arrow::Table> arrow_table;
  table->ToArrowTable(arrow_table);

  this->tables->push_back(arrow_table);

  auto const table_index = this->tables->size() - 1;
  for (int64_t i = 0; i < arrow_table->num_rows(); i++) {
    this->rows_set->insert(std::pair<int8_t, int64_t>(table_index, i));
  }
}

cylon::Status cylon::kernel::Union::Finalize(std::shared_ptr<cylon::Table> &result) {
  std::vector<std::vector<int64_t>> indices_from_tabs(this->tables->size());

  for (auto const &pr : *this->rows_set) {
    indices_from_tabs[pr.first].push_back(pr.second);
  }

  std::vector<std::shared_ptr<arrow::ChunkedArray>> final_data_arrays;
  // prepare final arrays
  for (int32_t col_idx = 0; col_idx < this->schema->num_fields(); col_idx++) {
    arrow::ArrayVector array_vector;
    for (size_t tab_idx = 0; tab_idx < this->tables->size(); tab_idx++) {
      auto status = cylon::kernel::PrepareArray(this->ctx,
                                                (*tables)[tab_idx],
                                                col_idx,
                                                indices_from_tabs[tab_idx],
                                                array_vector);

      if (!status.is_ok()) return status;
    }
    final_data_arrays.push_back(std::make_shared<arrow::ChunkedArray>(array_vector));
  }
  // create final table
  std::shared_ptr<arrow::Table> table = arrow::Table::Make(this->schema, final_data_arrays);
  auto merge_result = table->CombineChunks(cylon::ToArrowPool(this->ctx));
  const auto &merge_status = merge_result.status();
  if (!merge_status.ok()) {
    return Status(static_cast<int>(merge_status.code()), merge_status.message());
  }
  result = std::make_shared<cylon::Table>(ctx, merge_result.ValueOrDie());
  return cylon::Status::OK();
}

cylon::kernel::Union::Union(const std::shared_ptr<CylonContext> &ctx,
                            const std::shared_ptr<arrow::Schema> &schema,
                            int64_t expected_rows) :
    tables(std::make_shared<std::vector<std::shared_ptr<arrow::Table>>>()),
    schema(schema),
    ctx(ctx) {
  RowComparator row_comparator(tables, schema);
  rows_set =
      std::make_shared<std::unordered_set<std::pair<int8_t, int64_t>, RowComparator, RowComparator>>(expected_rows,
                                                                                                     row_comparator,
                                                                                                     row_comparator);
}

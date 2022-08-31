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

#include <string>
#include <vector>
#include <memory>
#include <set>

#include <cylon/join/join_utils.hpp>
#include <cylon/util/arrow_utils.hpp>
#include <cylon/util/macros.hpp>

namespace cylon {
namespace join {
namespace util {

std::shared_ptr<arrow::Schema> build_final_table_schema(const std::shared_ptr<arrow::Table> &left_tab,
                                                        const std::shared_ptr<arrow::Table> &right_tab,
                                                        const std::string &left_table_prefix,
                                                        const std::string &right_table_prefix) {
  // creating joined schema
  std::vector<std::shared_ptr<arrow::Field>> fields;

  std::unordered_map<std::string, int32_t> column_name_index;

  // adding left table
  for (const auto &field: left_tab->schema()->fields()) {
    column_name_index.insert(std::make_pair<>(field->name(), fields.size()));
    fields.emplace_back(field);
  }

  // adding right table
  for (const auto &field: right_tab->schema()->fields()) {
    auto new_field = field;
    if (column_name_index.find(field->name()) != column_name_index.end()) {
      // same column name exists in the left table
      // make the existing column name prefixed with left column prefix
      fields[column_name_index.find(field->name())->second] = field->WithName(left_table_prefix + field->name());

      // new field will be prefixed with the right table
      new_field = field->WithName(right_table_prefix + field->name());
    }
    // this is a unique column name
    column_name_index.insert(std::make_pair<>(new_field->name(), fields.size()));
    fields.emplace_back(new_field);
  }

  return arrow::schema(fields);
}

Status build_final_table_inplace_index(size_t left_inplace_column, size_t right_inplace_column,
                                       const std::vector<int64_t> &left_indices,
                                       const std::vector<int64_t> &right_indices,
                                       std::shared_ptr<arrow::UInt64Array> &left_index_sorted_column,
                                       std::shared_ptr<arrow::UInt64Array> &right_index_sorted_column,
                                       const std::shared_ptr<arrow::Table> &left_tab,
                                       const std::shared_ptr<arrow::Table> &right_tab,
                                       const std::string &left_table_prefix,
                                       const std::string &right_table_prefix,
                                       std::shared_ptr<arrow::Table> *final_table,
                                       arrow::MemoryPool *memory_pool) {
  const auto &schema = build_final_table_schema(left_tab, right_tab, left_table_prefix,
                                                right_table_prefix);

  std::vector<int64_t> indices_indexed;
  indices_indexed.reserve(left_indices.size());

  for (int64_t v: left_indices) {
    if (v < 0) {
      indices_indexed.push_back(v);
    } else {
      indices_indexed.push_back((int64_t) left_index_sorted_column->Value(v));
    }
  }
  left_index_sorted_column.reset();
  std::vector<std::shared_ptr<arrow::Array>> data_arrays;
  // build arrays for left tab
  const std::vector<std::shared_ptr<arrow::ChunkedArray>> &kVector = left_tab->columns();
  for (size_t i = 0; i < kVector.size(); i++) {
    std::shared_ptr<arrow::ChunkedArray> ca = kVector[i];
    std::shared_ptr<arrow::Array> destination_col_array;
    if (i == left_inplace_column) {
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(cylon::util::copy_array_by_indices(
          left_indices, cylon::util::GetChunkOrEmptyArray(ca, 0), &destination_col_array,
          memory_pool));
    } else {
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(cylon::util::copy_array_by_indices(
          indices_indexed, cylon::util::GetChunkOrEmptyArray(ca, 0), &destination_col_array,
          memory_pool));
    }
    data_arrays.push_back(destination_col_array);
  }

  indices_indexed.clear();
  indices_indexed.reserve(right_indices.size());
  for (int64_t v: right_indices) {
    if (v < 0) {
      indices_indexed.push_back(v);
    } else {
      indices_indexed.push_back((int64_t) right_index_sorted_column->Value(v));
    }
  }
  right_index_sorted_column.reset();
  // build arrays for right tab
  const std::vector<std::shared_ptr<arrow::ChunkedArray>> &rvector = right_tab->columns();
  for (size_t i = 0; i < rvector.size(); i++) {
    std::shared_ptr<arrow::ChunkedArray> ca = rvector[i];
    std::shared_ptr<arrow::Array> destination_col_array;
    if (i == right_inplace_column) {
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(cylon::util::copy_array_by_indices(
          right_indices, cylon::util::GetChunkOrEmptyArray(ca, 0), &destination_col_array,
          memory_pool));
    } else {
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(cylon::util::copy_array_by_indices(
          indices_indexed, cylon::util::GetChunkOrEmptyArray(ca, 0), &destination_col_array,
          memory_pool));
    }

    data_arrays.push_back(destination_col_array);
  }
  *final_table = arrow::Table::Make(schema, data_arrays);
  return Status::OK();
}

Status build_final_table(const std::vector<int64_t> &left_indices,
                         const std::vector<int64_t> &right_indices,
                         const std::shared_ptr<arrow::Table> &left_tab,
                         const std::shared_ptr<arrow::Table> &right_tab,
                         const std::string &left_table_prefix,
                         const std::string &right_table_prefix,
                         std::shared_ptr<arrow::Table> *final_table,
                         arrow::MemoryPool *pool) {
  const auto &schema =
      build_final_table_schema(left_tab, right_tab, left_table_prefix, right_table_prefix);

  std::vector<std::shared_ptr<arrow::ChunkedArray>> data_arrays;
  data_arrays.reserve(left_tab->num_columns() + right_tab->num_columns());

  // build arrays for left tab
  for (const auto &column: left_tab->columns()) {
    std::shared_ptr<arrow::Array> col_array;
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(
        cylon::util::copy_array_by_indices(left_indices,
                                           cylon::util::GetChunkOrEmptyArray(column, 0, pool),
                                           &col_array,
                                           pool));
    data_arrays.push_back(std::make_shared<arrow::ChunkedArray>(std::move(col_array)));
  }

  // build arrays for right tab
  for (const auto &column: right_tab->columns()) {
    std::shared_ptr<arrow::Array> col_array;
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(
        cylon::util::copy_array_by_indices(right_indices,
                                           cylon::util::GetChunkOrEmptyArray(column, 0),
                                           &col_array,
                                           pool));
    data_arrays.push_back(std::make_shared<arrow::ChunkedArray>(std::move(col_array)));
  }
  *final_table = arrow::Table::Make(schema, std::move(data_arrays));
  return Status::OK();
}

Status CombineChunks(const std::shared_ptr<arrow::Table> &table,
                     int col_index,
                     std::shared_ptr<arrow::Table> &output_table,
                     arrow::MemoryPool *memory_pool) {
  if (table->column(col_index)->num_chunks() > 1) {
    CYLON_ASSIGN_OR_RAISE(output_table, table->CombineChunks(memory_pool))
  } else {
    output_table = table;
  }
  return Status::OK();
}

}  // namespace util
}  // namespace join
}  // namespace cylon

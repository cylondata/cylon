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

#include "join_utils.hpp"
#include "../util/arrow_utils.hpp"

namespace cylon {
namespace join {
namespace util {

arrow::Status build_final_table_inplace_index(size_t left_inplace_column, size_t right_inplace_column,
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
  // creating joined schema
  std::vector<std::shared_ptr<arrow::Field>> fields;
  // TODO: get left and right suffixes from user if needed and update it here and replace in the schema with newfileds
  std::string  prefix = left_table_prefix;
  for(const auto &t: {left_tab, right_tab}){
    for (const auto &field: t->schema()->fields()){
      fields.emplace_back(field->WithName(prefix + field->name()));
    }
    prefix = right_table_prefix;
  }
  const auto &schema = arrow::schema(fields);

  std::vector<int64_t> indices_indexed;
  indices_indexed.reserve(left_indices.size());

  for (long v : left_indices) {
    if (v < 0){
      indices_indexed.push_back(v);
    } else {
      indices_indexed.push_back(left_index_sorted_column->Value(v));
    }
  }
  left_index_sorted_column.reset();
  std::vector<std::shared_ptr<arrow::Array>> data_arrays;
  // build arrays for left tab
  const std::vector<std::shared_ptr<arrow::ChunkedArray>> &kVector = left_tab->columns();
  for (size_t i = 0; i < kVector.size(); i++) {
    std::shared_ptr<arrow::ChunkedArray> ca = kVector[i];
    std::shared_ptr<arrow::Array> destination_col_array;
    arrow::Status status;
    if (i == left_inplace_column) {
      status = cylon::util::copy_array_by_indices(left_indices,
                                                  cylon::util::GetChunkOrEmptyArray(ca, 0),
                                                  &destination_col_array,
                                                  memory_pool);
    } else {
      status = cylon::util::copy_array_by_indices(indices_indexed,
                                                  cylon::util::GetChunkOrEmptyArray(ca, 0),
                                                  &destination_col_array,
                                                  memory_pool);
    }
    if (status != arrow::Status::OK()) {
      LOG(FATAL) << "Failed while copying a column to the final table from left table. "
                 << status.ToString();
      return status;
    }
    data_arrays.push_back(destination_col_array);
  }

  indices_indexed.clear();
  indices_indexed.reserve(right_indices.size());
  for (long v : right_indices) {
    if (v < 0){
      indices_indexed.push_back(v);
    } else {
      indices_indexed.push_back(right_index_sorted_column->Value(v));
    }
  }
  right_index_sorted_column.reset();
  // build arrays for right tab
  const std::vector<std::shared_ptr<arrow::ChunkedArray>> &rvector = right_tab->columns();
  for (size_t i = 0; i < rvector.size(); i++) {
    std::shared_ptr<arrow::ChunkedArray> ca = rvector[i];
    std::shared_ptr<arrow::Array> destination_col_array;
    arrow::Status status;
    if (i == right_inplace_column) {
      status = cylon::util::copy_array_by_indices(right_indices,
                                                  cylon::util::GetChunkOrEmptyArray(ca, 0),
                                                  &destination_col_array,
                                                  memory_pool);
    } else {
      status = cylon::util::copy_array_by_indices(indices_indexed,
                                                  cylon::util::GetChunkOrEmptyArray(ca, 0),
                                                  &destination_col_array,
                                                  memory_pool);
    }
    if (status != arrow::Status::OK()) {
      LOG(FATAL) << "Failed while copying a column to the final table from right table. "
                 << status.ToString();
      return status;
    }
    data_arrays.push_back(destination_col_array);
  }
  *final_table = arrow::Table::Make(schema, data_arrays);
  return arrow::Status::OK();
}

arrow::Status build_final_table(const std::vector<int64_t> &left_indices,
                                const std::vector<int64_t> &right_indices,
                                const std::shared_ptr<arrow::Table> &left_tab,
                                const std::shared_ptr<arrow::Table> &right_tab,
                                const std::string &left_table_prefix,
                                const std::string &right_table_prefix,
                                std::shared_ptr<arrow::Table> *final_table,
                                arrow::MemoryPool *memory_pool) {
  // creating joined schema
  std::vector<std::shared_ptr<arrow::Field>> fields;
  // TODO: get left and right suffixes from user if needed and update it here and replace in the schema with newfileds
  std::string  prefix = left_table_prefix;
  for(const auto &t: {left_tab, right_tab}){
    for (const auto &field: t->schema()->fields()){
      fields.emplace_back(field->WithName(prefix + field->name()));
    }
    prefix = right_table_prefix;
  }
  const auto &schema = arrow::schema(fields);

  std::vector<std::shared_ptr<arrow::Array>> data_arrays;

  // build arrays for left tab
  for (auto &column : left_tab->columns()) {
    std::shared_ptr<arrow::Array> destination_col_array;
    arrow::Status
        status = cylon::util::copy_array_by_indices(left_indices,
                                                    cylon::util::GetChunkOrEmptyArray(column, 0),
                                                    &destination_col_array,
                                                    memory_pool);
    if (status != arrow::Status::OK()) {
      LOG(FATAL) << "Failed while copying a column to the final table from left table. "
                 << status.ToString();
      return status;
    }
    data_arrays.push_back(destination_col_array);
  }

  // build arrays for right tab
  for (auto &column : right_tab->columns()) {
    std::shared_ptr<arrow::Array> destination_col_array;
    arrow::Status
        status = cylon::util::copy_array_by_indices(right_indices,
                                                    cylon::util::GetChunkOrEmptyArray(column, 0),
                                                    &destination_col_array,
                                                    memory_pool);
    if (status != arrow::Status::OK()) {
      LOG(FATAL) << "Failed while copying a column to the final table from right table. "
                 << status.ToString();
      return status;
    }
    data_arrays.push_back(destination_col_array);
  }
  *final_table = arrow::Table::Make(schema, data_arrays);
  return arrow::Status::OK();
}

arrow::Status CombineChunks(const std::shared_ptr<arrow::Table> &table,
                            int col_index,
                            std::shared_ptr<arrow::Table> &output_table,
                            arrow::MemoryPool *memory_pool) {
  if (table->column(col_index)->num_chunks() > 1) {
    LOG(INFO) << "Combining chunks " << table->column(col_index)->num_chunks();
    arrow::Result<std::shared_ptr<arrow::Table>> result = table->CombineChunks(memory_pool);
    if (result.ok()) {
      output_table = result.ValueOrDie();
    }
    return result.status();
  } else {
    output_table = table;
    return arrow::Status::OK();
  }
}

}  // namespace util
}  // namespace join
}  // namespace cylon

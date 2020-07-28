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

arrow::Status build_final_table(const std::shared_ptr<std::vector<int64_t>> &left_indices,
                                const std::shared_ptr<std::vector<int64_t>> &right_indices,
                                const std::shared_ptr<arrow::Table> &left_tab,
                                const std::shared_ptr<arrow::Table> &right_tab,
                                std::shared_ptr<arrow::Table> *final_table,
                                arrow::MemoryPool *memory_pool) {
  // creating joined schema
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::Field>> new_fields;
  uint64_t left_table_columns = left_tab->schema()->num_fields();
  fields.insert(fields.end(), left_tab->schema()->fields().begin(),
                left_tab->schema()->fields().end());
  fields.insert(fields.end(), right_tab->schema()->fields().begin(),
                right_tab->schema()->fields().end());
  std::string left_tb_prefix = "lt-";
  std::string right_tb_prefix = "rt-";
  std::string prefix = left_tb_prefix;
  for (size_t i = 0; i < fields.size(); i++) {
    if (i >= left_table_columns) {
      prefix = right_tb_prefix;
    }
    new_fields.push_back(std::make_shared<arrow::Field>(prefix + std::to_string(i),
                                                        fields.at(i)->type(),
                                                        fields.at(i)->nullable()));
  }
  auto schema = arrow::schema(new_fields);

  std::vector<std::shared_ptr<arrow::Array>> data_arrays;

  // build arrays for left tab
  for (auto &column : left_tab->columns()) {
    std::shared_ptr<arrow::Array> destination_col_array;
    arrow::Status
        status = cylon::util::copy_array_by_indices(left_indices,
                                                    column->chunk(0),
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
                                                    column->chunk(0),
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
                            int64_t col_index,
                            std::shared_ptr<arrow::Table> &output_table,
                            arrow::MemoryPool *memory_pool) {
  if (table->column(col_index)->num_chunks() > 1) {
    LOG(INFO) << "Combining chunks " << table->column(col_index)->num_chunks();
    return table->CombineChunks(memory_pool, &output_table);
  } else {
    output_table = table;
    return arrow::Status::OK();
  }
}

}  // namespace util
}  // namespace join
}  // namespace cylon

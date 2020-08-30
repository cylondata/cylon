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
#include <arrow/arrow_partition_kernels.hpp>
#include "partition.hpp"

namespace cylon {
namespace kernel {
Status HashPartition(cylon::CylonContext *ctx, const std::shared_ptr<cylon::Table> table,
                     const std::vector<int> &hash_columns, int no_of_partitions,
                     std::unordered_map<int, std::shared_ptr<cylon::Table>> *out) {
  std::shared_ptr<arrow::Table> table_;
  table->ToArrowTable(table_);

  // keep arrays for each target, these arrays are used for creating the table
  std::unordered_map<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>> data_arrays;
  std::vector<int> partitions;
  for (int t = 0; t < no_of_partitions; t++) {
    partitions.push_back(t);
    data_arrays.insert(
        std::pair<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>>(
            t, std::make_shared<std::vector<std::shared_ptr<arrow::Array>>>()));
  }

  std::vector<std::shared_ptr<arrow::Array>> arrays;
  int64_t length = 0;
  for (auto col_index : hash_columns) {
    auto column = table_->column(col_index);
    std::shared_ptr<arrow::Array> array = column->chunk(0);
    arrays.push_back(array);

    if (!(length == 0 || length == column->length())) {
      return Status(cylon::IndexError,
                    "Column lengths doesnt match " + std::to_string(length));
    }
    length = column->length();
  }

  // first we partition the table
  std::vector<int64_t> outPartitions;
  outPartitions.reserve(length);
  Status status = HashPartitionArrays(cylon::ToArrowPool(ctx), arrays, length,
                                      partitions, &outPartitions);
  if (!status.is_ok()) {
    LOG(FATAL) << "Failed to create the hash partition";
    return status;
  }

  for (int i = 0; i < table_->num_columns(); i++) {
    std::shared_ptr<arrow::DataType> type = table_->column(i)->chunk(0)->type();
    std::shared_ptr<arrow::Array> array = table_->column(i)->chunk(0);

    std::shared_ptr<ArrowArraySplitKernel> splitKernel;
    status = CreateSplitter(type, cylon::ToArrowPool(ctx), &splitKernel);
    if (!status.is_ok()) {
      LOG(FATAL) << "Failed to create the splitter";
      return status;
    }

    // this one outputs arrays for each target as a map
    std::unordered_map<int, std::shared_ptr<arrow::Array>> splited_arrays;
    splitKernel->Split(array, outPartitions, partitions, splited_arrays);

    for (const auto &x : splited_arrays) {
      std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>> cols = data_arrays[x.first];
      cols->push_back(x.second);
    }
  }
  // now insert these array to
  for (const auto &x : data_arrays) {
    std::shared_ptr<arrow::Table> final_arrow_table = arrow::Table::Make(table_->schema(), *x.second);
    std::shared_ptr<cylon::Table> kY = std::make_shared<cylon::Table>(final_arrow_table, ctx);
    out->insert(std::pair<int, std::shared_ptr<cylon::Table>>(x.first, kY));
  }
  return Status::OK();
}
}
}
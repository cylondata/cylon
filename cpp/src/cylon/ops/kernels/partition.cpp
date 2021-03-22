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
#include <vector>
#include <util/macros.hpp>
#include <util/arrow_utils.hpp>

#include "ctx/arrow_memory_pool_utils.hpp"
#include "partition.hpp"

cylon::kernel::StreamingHashPartitionKernel::StreamingHashPartitionKernel(const std::shared_ptr<cylon::CylonContext> &ctx,
                                                                          const std::shared_ptr<arrow::Schema> &schema,
                                                                          int num_partitions,
                                                                          const std::vector<int> &hash_columns)
    : num_partitions(num_partitions), hash_columns(hash_columns), schema(schema), ctx(ctx) {
  partition_kernels.reserve(hash_columns.size());
  for (auto &&col:hash_columns) {
    partition_kernels.emplace_back(CreateHashPartitionKernel(schema->field(col)->type()));
  }

  split_kernels.reserve(schema->num_fields());
  for (auto &&field:schema->fields()) {
    split_kernels.emplace_back(CreateStreamingSplitter(field->type(),
                                                       num_partitions,
                                                       cylon::ToArrowPool(ctx)));
  }

  temp_partition_hist.resize(num_partitions, 0);
}

cylon::Status cylon::kernel::StreamingHashPartitionKernel::Process(int tag, const std::shared_ptr<Table> &table) {
  const std::shared_ptr<arrow::Table> &arrow_table = table->get_table();

  if (arrow_table->column(0)->num_chunks() > 1) {
    return Status(Code::Invalid, "chunked arrays not supported"); // todo check this!
  }

  // first resize the vectors for partitions and counts
  temp_partitions.resize(arrow_table->num_rows());
  std::fill(temp_partitions.begin(), temp_partitions.end(), 0);
  std::fill(temp_partition_hist.begin(), temp_partition_hist.end(), 0);

  // building hash without the last hash_column_idx
  for (size_t i = 0; i < hash_columns.size() - 1; i++) {
    RETURN_CYLON_STATUS_IF_FAILED(partition_kernels[i]
                                      ->UpdateHash(arrow_table->column(hash_columns[i]), temp_partitions));
  }

  // build hash from the last hash_column_idx
  RETURN_CYLON_STATUS_IF_FAILED(partition_kernels.back()->Partition(
      arrow_table->column(hash_columns.back()),
      num_partitions,
      temp_partitions,
      temp_partition_hist));

  // now insert table to split_kernels
  for (int i = 0; i < arrow_table->num_columns(); i++) {
    RETURN_CYLON_STATUS_IF_FAILED(split_kernels[i]->Split(
        cylon::util::GetChunkOrEmptyArray(arrow_table->column(i), 0), temp_partitions, temp_partition_hist));
  }

  return Status::OK();
}

cylon::Status cylon::kernel::StreamingHashPartitionKernel::Finish(std::vector<std::shared_ptr<Table>> &partitioned_tables) {
  // no longer needs the temp arrays
  temp_partitions.clear();
  temp_partition_hist.clear();

  // call streaming split kernel finalize and create the final arrays
  std::vector<arrow::ArrayVector> data_arrays(num_partitions); // size num_partitions

  std::vector<std::shared_ptr<arrow::Array>> temp_split_arrays;
  for (const auto &splitKernel:split_kernels) {
    RETURN_CYLON_STATUS_IF_FAILED(splitKernel->Finish(temp_split_arrays));
    for (size_t i = 0; i < temp_split_arrays.size(); i++) {
      // remove the array and put it in the data arrays vector
      data_arrays[i].emplace_back(std::move(temp_split_arrays[i]));
    }
    temp_split_arrays.clear();
  }

  partitioned_tables.reserve(num_partitions);
  for (const auto &arr_vec: data_arrays) {
    auto a_table = arrow::Table::Make(schema, arr_vec);
    partitioned_tables.emplace_back(std::make_shared<Table>(a_table, ctx));
  }

  return cylon::Status::OK();
}

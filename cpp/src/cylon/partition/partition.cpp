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
#include <chrono>

#include "../arrow/arrow_kernels.hpp"
#include "../ctx/arrow_memory_pool_utils.hpp"
#include "../util/macros.hpp"
#include "../arrow/arrow_partition_kernels.hpp"

#include "partition.hpp"

namespace cylon {

static inline Status split_impl(const std::shared_ptr<Table> &table,
                                uint32_t num_partitions,
                                const std::vector<uint32_t> &target_partitions,
                                const std::vector<uint32_t> &partition_hist,
                                std::vector<std::shared_ptr<arrow::Table>> &output) {
  auto t1 = std::chrono::high_resolution_clock::now();
  Status status;
  const std::shared_ptr<arrow::Table> &arrow_table = table->get_table();
  std::shared_ptr<cylon::CylonContext> ctx = table->GetContext();
  arrow::MemoryPool *pool = cylon::ToArrowPool(ctx);

  std::vector<arrow::ArrayVector> data_arrays(num_partitions); // size num_partitions

  for (const auto &col:arrow_table->columns()) {
    std::unique_ptr<ArrowArraySplitKernel> splitKernel = CreateSplitter(col->type(), pool);
    if (splitKernel == nullptr) return Status(Code::NotImplemented, "splitter not implemented");

    std::vector<std::shared_ptr<arrow::Array>> split_arrays;
    RETURN_CYLON_STATUS_IF_FAILED(splitKernel
                                      ->Split(col, num_partitions, target_partitions, partition_hist, split_arrays));

    for (size_t i = 0; i < split_arrays.size(); i++) {
      data_arrays[i].push_back(split_arrays[i]);
    }
  }

  output.reserve(num_partitions);
  for (const auto &arr_vec: data_arrays) {
    output.push_back(arrow::Table::Make(arrow_table->schema(), arr_vec));
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Splitting table time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  return Status::OK();
}


Status Split(const std::shared_ptr<Table> &table,
             uint32_t num_partitions,
             const std::vector<uint32_t> &target_partitions,
             std::vector<std::shared_ptr<arrow::Table>> &output) {
  if ((size_t) table->Rows() != target_partitions.size()) {
    LOG_AND_RETURN_ERROR(Code::ExecutionError, "tables rows != target_partitions length");
  }
  std::vector<uint32_t> partition_hist(num_partitions, 0);
  for (const int32_t p:target_partitions) {
    partition_hist[p]++;
  }
  return split_impl(table, num_partitions, target_partitions, partition_hist, output);
}


Status Split(const std::shared_ptr<Table> &table,
             uint32_t num_partitions,
             const std::vector<uint32_t> &target_partitions,
             const std::vector<uint32_t> &partition_hist_ptr,
             std::vector<std::shared_ptr<arrow::Table>> &output) {
  if ((size_t) table->Rows() != target_partitions.size()) {
    LOG_AND_RETURN_ERROR(Code::ExecutionError, "tables rows != target_partitions length");
  }
  return split_impl(table, num_partitions, target_partitions, partition_hist_ptr, output);
}

Status MapToHashPartitions(const std::shared_ptr<Table> &table,
                           int32_t hash_column_idx,
                           uint32_t num_partitions,
                           std::vector<uint32_t> &target_partitions,
                           std::vector<uint32_t> &partition_hist) {
  auto t1 = std::chrono::high_resolution_clock::now();
  const std::shared_ptr<arrow::Table> &arrow_table = table->get_table();
  std::shared_ptr<arrow::ChunkedArray> idx_col = arrow_table->column(hash_column_idx);

  std::unique_ptr<PartitionKernel> kern = CreateHashPartitionKernel(idx_col->type());
  if (kern == nullptr) {
    LOG_AND_RETURN_ERROR(Code::ExecutionError, "unable to create hash partition kernel");
  }
  // initialize vectors
  std::fill(target_partitions.begin(), target_partitions.end(), 0);
  std::fill(partition_hist.begin(), partition_hist.end(), 0);
  target_partitions.resize(idx_col->length(), 0);
  partition_hist.resize(num_partitions, 0);

  Status status = kern->Partition(idx_col, num_partitions, target_partitions, partition_hist);

  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Hash partition time : " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  return status;
}

Status MapToHashPartitions(const std::shared_ptr<Table> &table,
                           const std::vector<int32_t> &hash_column_idx,
                           uint32_t num_partitions,
                           std::vector<uint32_t> &target_partitions,
                           std::vector<uint32_t> &partition_hist) {
  auto t1 = std::chrono::high_resolution_clock::now();
  Status status;
  const std::shared_ptr<arrow::Table> &arrow_table = table->get_table();
  std::shared_ptr<cylon::CylonContext> ctx = table->GetContext();

  std::vector<std::unique_ptr<HashPartitionKernel>> partition_kernels;
  const std::vector<std::shared_ptr<arrow::Field>> &fields = arrow_table->schema()->fields();
  for (int i : hash_column_idx) {
    const std::shared_ptr<arrow::DataType> &type = fields[i]->type();
    auto kern = CreateHashPartitionKernel(type);
    if (kern == nullptr) {
      LOG_AND_RETURN_ERROR(Code::ExecutionError, "unable to create range partition kernel");
    }
    partition_kernels.push_back(std::move(kern));
  }

  // initialize vectors
  std::fill(target_partitions.begin(), target_partitions.end(), 0);
  target_partitions.resize(arrow_table->num_rows(), 0);
  std::fill(partition_hist.begin(), partition_hist.end(), 0);
  partition_hist.resize(num_partitions, 0);

  // building hash without the last hash_column_idx
  for (size_t i = 0; i < hash_column_idx.size() - 1; i++) {
    auto t11 = std::chrono::high_resolution_clock::now();

    RETURN_CYLON_STATUS_IF_FAILED(partition_kernels[i]->UpdateHash(arrow_table->column(hash_column_idx[i]),
                                                                   target_partitions));
    auto t12 = std::chrono::high_resolution_clock::now();

    LOG(INFO) << "building hash (idx " << hash_column_idx[i] << ") time : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t11).count();
  }

  // build hash from the last hash_column_idx
  status = partition_kernels.back()->Partition(arrow_table->column(hash_column_idx.back()),
                                               num_partitions,
                                               target_partitions,
                                               partition_hist);

  const auto &t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Partition time : " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  return status;
}

Status MapToSortPartitions(const std::shared_ptr<Table> &table,
                           int32_t column_idx,
                           uint32_t num_partitions,
                           std::vector<uint32_t> &target_partitions,
                           std::vector<uint32_t> &partition_hist,
                           bool ascending,
                           uint64_t num_samples,
                           uint32_t num_bins) {
  auto t1 = std::chrono::high_resolution_clock::now();
  std::shared_ptr<CylonContext> ctx = table->GetContext();
  const std::shared_ptr<arrow::Table> &arrow_table = table->get_table();
  std::shared_ptr<arrow::ChunkedArray> idx_col = arrow_table->column(column_idx);

  if (num_samples == 0) num_samples = std::min((int64_t) (table->Rows() * 0.01), table->Rows());
  if (num_bins == 0) num_bins = num_partitions * 16;

  std::unique_ptr<PartitionKernel> kern = CreateRangePartitionKernel(idx_col->type(),
                                                                     ctx, ascending, num_samples, num_bins);
  if (kern == nullptr) {
    LOG_AND_RETURN_ERROR(Code::ExecutionError, "unable to create range partition kernel");
  }

  // initialize vectors
  target_partitions.resize(idx_col->length(), 0);
  partition_hist.resize(num_partitions, 0);

  const auto &status = kern->Partition(idx_col, num_partitions, target_partitions, partition_hist);
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Sort partition time : " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  return status;
}

Status PartitionByHashing(const std::shared_ptr<Table> &table,
                          const std::vector<int32_t> &hash_cols,
                          uint32_t num_partitions,
                          std::vector<std::shared_ptr<Table>> &partitions) {
  // partition the tables locally
  std::vector<uint32_t> outPartitions, counts;
  LOG_AND_RETURN_CYLON_STATUS_IF_FAILED(MapToHashPartitions(table, hash_cols, num_partitions, outPartitions, counts));

  std::vector<std::shared_ptr<arrow::Table>> partitioned_tables;
  LOG_AND_RETURN_CYLON_STATUS_IF_FAILED(split_impl(table, num_partitions, outPartitions, counts, partitioned_tables));

  partitions.reserve(num_partitions);
  std::shared_ptr<cylon::CylonContext> ctx = table->GetContext();
  for (auto &&atable:partitioned_tables) {
    partitions.emplace_back(std::make_shared<Table>(atable, ctx));
  }
  return Status::OK();
}

}
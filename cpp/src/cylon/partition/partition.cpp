//
// Created by niranda on 11/19/20.
//

#include <glog/logging.h>
#include <ctx/arrow_memory_pool_utils.hpp>

#include "partition.hpp"
#include "util/macros.hpp"
#include "arrow/arrow_partition_kernels.hpp"

namespace cylon {

static Status split_impl(const std::shared_ptr<Table> &table,
                         const std::vector<uint32_t> &target_partitions,
                         uint32_t num_partitions,
                         std::vector<std::shared_ptr<Table>> &output,
                         const std::vector<uint32_t> *partition_hist_ptr) {
  auto t1 = std::chrono::high_resolution_clock::now();
  Status status;
  const std::shared_ptr<arrow::Table> &arrow_table = table->get_table();
  std::shared_ptr<cylon::CylonContext> ctx = table->GetContext();
  arrow::MemoryPool *pool = cylon::ToArrowPool(ctx);

  std::vector<arrow::ArrayVector> data_arrays(num_partitions); // size num_partitions

  for (const auto &col:arrow_table->columns()) {
    std::shared_ptr<ArrowArraySplitKernel> splitKernel;
    status = CreateSplitter(col->type(), pool, &splitKernel);
    RETURN_CYLON_STATUS_IF_FAILED(status)

    std::vector<std::shared_ptr<arrow::Array>> split_arrays;
    status = splitKernel->Split(col, target_partitions, num_partitions, *partition_hist_ptr, split_arrays);
    RETURN_CYLON_STATUS_IF_FAILED(status)

    for (size_t i = 0; i < split_arrays.size(); i++) {
      data_arrays[i].push_back(split_arrays[i]);
    }
  }

  output.reserve(num_partitions);
  for (const auto &arr_vec: data_arrays) {
    std::shared_ptr<arrow::Table> arrow_table_out = arrow::Table::Make(arrow_table->schema(), arr_vec);
    std::shared_ptr<Table> cylon_table_out;
    status = cylon::Table::FromArrowTable(ctx, arrow_table_out, cylon_table_out);
    RETURN_CYLON_STATUS_IF_FAILED(status)
    output.push_back(std::move(cylon_table_out));
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Splitting table time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  return Status::OK();
}

Status Split(const std::shared_ptr<Table> &table,
             const std::vector<uint32_t> &target_partitions,
             uint32_t num_partitions,
             std::vector<std::shared_ptr<Table>> &output,
             const std::vector<uint32_t> *partition_hist_ptr) {

  if ((size_t) table->Rows() != target_partitions.size()) {
    LOG_AND_RETURN_ERROR(Code::ExecutionError, "tables rows != target_partitions length")
  }

  if (partition_hist_ptr == nullptr) {
    LOG(INFO) << "building partition histogram";
    std::vector<uint32_t> partition_hist(num_partitions, 0);
    for (const int32_t p:target_partitions) {
      partition_hist[p]++;
    }

    return split_impl(table, target_partitions, num_partitions, output, &partition_hist);
  } else {
    return split_impl(table, target_partitions, num_partitions, output, partition_hist_ptr);
  }
}

static inline int32_t modulus(const uint64_t val, const uint32_t div) {
  return val % div;
}

static inline int32_t modulus_div_power2(const uint64_t val, const uint32_t div) {
  return val & (div - 1);
}

Status ModuloPartition(const std::shared_ptr<Table> &table,
                       int32_t hash_column_idx,
                       uint32_t num_partitions,
                       std::vector<uint32_t> &target_partitions,
                       std::vector<uint32_t> &partition_hist) {
  auto t1 = std::chrono::high_resolution_clock::now();
  const std::shared_ptr<arrow::Table> &arrow_table = table->get_table();
  std::shared_ptr<arrow::ChunkedArray> idx_col = arrow_table->column(hash_column_idx);

  std::unique_ptr<ArrowPartitionKernel2> kern;
  switch (idx_col->type()->id()) {
    case arrow::Type::BOOL:kern = std::make_unique<ModuloPartitionKernel<arrow::BooleanType>>(num_partitions);
      break;
    case arrow::Type::UINT8:kern = std::make_unique<ModuloPartitionKernel<arrow::UInt8Type>>(num_partitions);
      break;
    case arrow::Type::INT8:kern = std::make_unique<ModuloPartitionKernel<arrow::Int8Type>>(num_partitions);
      break;
    case arrow::Type::UINT16:kern = std::make_unique<ModuloPartitionKernel<arrow::UInt16Type>>(num_partitions);
      break;
    case arrow::Type::INT16:kern = std::make_unique<ModuloPartitionKernel<arrow::Int16Type>>(num_partitions);
      break;
    case arrow::Type::UINT32:kern = std::make_unique<ModuloPartitionKernel<arrow::UInt32Type>>(num_partitions);
      break;
    case arrow::Type::INT32:kern = std::make_unique<ModuloPartitionKernel<arrow::Int32Type>>(num_partitions);
      break;
    case arrow::Type::UINT64:kern = std::make_unique<ModuloPartitionKernel<arrow::UInt64Type>>(num_partitions);
      break;
    case arrow::Type::INT64:kern = std::make_unique<ModuloPartitionKernel<arrow::Int64Type>>(num_partitions);
      break;
    default:LOG_AND_RETURN_ERROR(Code::Invalid, "modulo partition works only for integer values")
  }

  const Status &status = kern->Partition(idx_col, target_partitions, partition_hist);
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Modulo partition time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  return status;
}

Status HashPartition(const std::shared_ptr<Table> &table,
                     int32_t hash_column_idx,
                     uint32_t num_partitions,
                     std::vector<uint32_t> &target_partitions,
                     std::vector<uint32_t> &partition_hist) {
  auto t1 = std::chrono::high_resolution_clock::now();
  const std::shared_ptr<arrow::Table> &arrow_table = table->get_table();
  std::shared_ptr<arrow::ChunkedArray> idx_col = arrow_table->column(hash_column_idx);

  std::unique_ptr<ArrowPartitionKernel2> kern;
  Status status = CreateHashPartitionKernel(idx_col->type(), num_partitions, kern);
  RETURN_CYLON_STATUS_IF_FAILED(status)

  // initialize vectors
  std::fill(target_partitions.begin(), target_partitions.end(), 0);
  std::fill(partition_hist.begin(), partition_hist.end(), 0);
  target_partitions.resize(idx_col->length(), 0);
  partition_hist.resize(num_partitions, 0);

  status = kern->Partition(idx_col, target_partitions, partition_hist);
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Modulo partition time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  return status;
}

Status ApplyPartition(const std::shared_ptr<Table> &table,
                      const std::vector<int32_t> &hash_column_idx,
                      uint32_t num_partitions,
                      std::vector<uint32_t> &target_partitions,
                      std::vector<uint32_t> &partition_hist) {

  auto t1 = std::chrono::high_resolution_clock::now();
  Status status;
  const std::shared_ptr<arrow::Table> &arrow_table = table->get_table();
  std::shared_ptr<cylon::CylonContext> ctx = table->GetContext();

  std::vector<std::unique_ptr<ArrowPartitionKernel2>> partition_kernels(hash_column_idx.size());
  const std::vector<std::shared_ptr<arrow::Field>> &fields = arrow_table->schema()->fields();
  for (size_t i = 0; i < hash_column_idx.size(); i++) {
    const std::shared_ptr<arrow::DataType> &type = fields[hash_column_idx[i]]->type();

    status = CreateHashPartitionKernel(type, num_partitions, partition_kernels[i]);
    RETURN_CYLON_STATUS_IF_FAILED(status)
  }

  // initialize vectors
  std::fill(target_partitions.begin(), target_partitions.end(), 0);
  target_partitions.resize(arrow_table->num_rows(), 0);
  std::fill(partition_hist.begin(), partition_hist.end(), 0);
  partition_hist.resize(num_partitions, 0);

  // building hash without the last hash_column_idx
  for (size_t i = 0; i < hash_column_idx.size() - 1; i++) {
    auto t11 = std::chrono::high_resolution_clock::now();
    status =
        partition_kernels[i]->BuildHash(arrow_table->column(hash_column_idx[i]), target_partitions);
    auto t12 = std::chrono::high_resolution_clock::now();

    LOG(INFO) << "building hash (idx " << hash_column_idx[i] << ") time : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t11).count();
    RETURN_CYLON_STATUS_IF_FAILED(status)
  }

  // build hash from the last hash_column_idx
  status = partition_kernels.back()->Partition(arrow_table->column(hash_column_idx.back()),
                                               target_partitions, partition_hist);
  RETURN_CYLON_STATUS_IF_FAILED(status)
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Partition time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  return Status::OK();
}

}
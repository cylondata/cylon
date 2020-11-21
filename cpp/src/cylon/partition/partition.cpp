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
    RETURN_IF_STATUS_FAILED(status)

    std::vector<std::shared_ptr<arrow::Array>> split_arrays;
    status = splitKernel->Split(col, target_partitions, num_partitions, *partition_hist_ptr, split_arrays);
    RETURN_IF_STATUS_FAILED(status)

    for (size_t i = 0; i < split_arrays.size(); i++) {
      data_arrays[i].push_back(split_arrays[i]);
    }
  }

  output.reserve(num_partitions);
  for (const auto &arr_vec: data_arrays) {
    std::shared_ptr<arrow::Table> arrow_table_out = arrow::Table::Make(arrow_table->schema(), arr_vec);
    std::shared_ptr<Table> cylon_table_out;
    status = cylon::Table::FromArrowTable(ctx, arrow_table_out, cylon_table_out);
    RETURN_IF_STATUS_FAILED(status)
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
//
//template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
//static inline bool if_power2(T v) {
//  return v && !(v & (v - 1));
//}

/*template<typename TYPE>
static Status modulo_parition_impl(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                                   int32_t num_partitions,
                                   std::vector<int32_t> &target_partitions,
                                   std::vector<uint32_t> &partition_histogram) {
  using ARROW_ARRAY_T = typename arrow::TypeTraits<TYPE>::ArrayType;

  auto t1 = std::chrono::high_resolution_clock::now();

  if (!partition_histogram.empty() || !target_partitions.empty()) {
    return Status(Code::Invalid, "target partitions or histogram not empty!");
  }

  // initialize the histogram
  partition_histogram.reserve(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    partition_histogram.push_back(0);
  }

  std::function<int32_t(const uint64_t, const uint32_t)>
      fptr = if_power2(num_partitions) ? &modulus_div_power2 : &modulus;

  target_partitions.reserve(idx_col->length());
  for (const auto &arr: idx_col->chunks()) {
    const std::shared_ptr<ARROW_ARRAY_T> &carr = std::static_pointer_cast<ARROW_ARRAY_T>(arr);
    for (int64_t i = 0; i < carr->length(); i++) {
      int32_t p = fptr(carr->Value(i), num_partitions);
      target_partitions.push_back(p);
      partition_histogram[p]++;
    }
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Modulo partition time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  return Status::OK();
}*/

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
  switch (idx_col->type()->id()) {
    case arrow::Type::BOOL:kern = std::make_unique<HashPartitionKernel<arrow::BooleanType>>(num_partitions);
      break;
    case arrow::Type::UINT8:kern = std::make_unique<HashPartitionKernel<arrow::UInt8Type>>(num_partitions);
      break;
    case arrow::Type::INT8:kern = std::make_unique<HashPartitionKernel<arrow::Int8Type>>(num_partitions);
      break;
    case arrow::Type::UINT16:kern = std::make_unique<HashPartitionKernel<arrow::UInt16Type>>(num_partitions);
      break;
    case arrow::Type::INT16:kern = std::make_unique<HashPartitionKernel<arrow::Int16Type>>(num_partitions);
      break;
    case arrow::Type::UINT32:kern = std::make_unique<HashPartitionKernel<arrow::UInt32Type>>(num_partitions);
      break;
    case arrow::Type::INT32:kern = std::make_unique<HashPartitionKernel<arrow::Int32Type>>(num_partitions);
      break;
    case arrow::Type::UINT64:kern = std::make_unique<HashPartitionKernel<arrow::UInt64Type>>(num_partitions);
      break;
    case arrow::Type::INT64:kern = std::make_unique<HashPartitionKernel<arrow::Int64Type>>(num_partitions);
      break;
    case arrow::Type::FLOAT:kern = std::make_unique<HashPartitionKernel<arrow::FloatType>>(num_partitions);
      break;
    case arrow::Type::DOUBLE:kern = std::make_unique<HashPartitionKernel<arrow::DoubleType>>(num_partitions);
      break;
    default:LOG_AND_RETURN_ERROR(Code::Invalid, "modulo partition works only for integer values")
  }

  const Status &status = kern->Partition(idx_col, target_partitions, partition_hist);
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Modulo partition time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  return status;
}

Status ModuloPartition(const std::shared_ptr<Table> &table,
                       std::vector<int32_t> &hash_column_idx,
                       int32_t num_partitions,
                       std::vector<int32_t> &target_partitions,
                       std::vector<uint32_t> &partition_histogram) {
  return Status();
}

}


/*
     case arrow::Type::BOOL:
      return modulo_parition_impl<arrow::BooleanType>(idx_col,
                                                      num_partitions,
                                                      target_partitions,
                                                      partition_hist);
    case arrow::Type::UINT8:
      return modulo_parition_impl<arrow::UInt8Type>(idx_col,
                                                    num_partitions,
                                                    target_partitions,
                                                    partition_hist);

    case arrow::Type::INT8:
      return modulo_parition_impl<arrow::Int8Type>(idx_col,
                                                   num_partitions,
                                                   target_partitions,
                                                   partition_hist);
    case arrow::Type::UINT16:
      return modulo_parition_impl<arrow::UInt16Type>(idx_col,
                                                     num_partitions,
                                                     target_partitions,
                                                     partition_hist);
    case arrow::Type::INT16:
      return modulo_parition_impl<arrow::Int16Type>(idx_col,
                                                    num_partitions,
                                                    target_partitions,
                                                    partition_hist);
    case arrow::Type::UINT32:
      return modulo_parition_impl<arrow::UInt32Type>(idx_col,
                                                     num_partitions,
                                                     target_partitions,
                                                     partition_hist);
    case arrow::Type::INT32:
      return modulo_parition_impl<arrow::Int32Type>(idx_col,
                                                    num_partitions,
                                                    target_partitions,
                                                    partition_hist);
    case arrow::Type::UINT64:
      return modulo_parition_impl<arrow::UInt64Type>(idx_col,
                                                     num_partitions,
                                                     target_partitions,
                                                     partition_hist);
    case arrow::Type::INT64:
      return modulo_parition_impl<arrow::Int64Type>(idx_col,
                                                    num_partitions,
                                                    target_partitions,
                                                    partition_hist);
 */
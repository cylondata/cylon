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

#include "arrow_partition_kernels.hpp"

namespace cylon {

std::shared_ptr<ArrowPartitionKernel> GetPartitionKernel(arrow::MemoryPool *pool,
                                                         const std::shared_ptr<arrow::DataType> &data_type) {
  std::shared_ptr<ArrowPartitionKernel> kernel;
  switch (data_type->id()) {
    case arrow::Type::UINT8:return std::make_shared<UInt8ArrayHashPartitioner>(pool);
    case arrow::Type::INT8:return std::make_shared<Int8ArrayHashPartitioner>(pool);
    case arrow::Type::UINT16:return std::make_shared<UInt16ArrayHashPartitioner>(pool);
    case arrow::Type::INT16:return std::make_shared<Int16ArrayHashPartitioner>(pool);
    case arrow::Type::UINT32:return std::make_shared<UInt32ArrayHashPartitioner>(pool);
    case arrow::Type::INT32:return std::make_shared<Int32ArrayHashPartitioner>(pool);
    case arrow::Type::UINT64:return std::make_shared<UInt64ArrayHashPartitioner>(pool);
    case arrow::Type::INT64:return std::make_shared<Int64ArrayHashPartitioner>(pool);
    case arrow::Type::FLOAT:return std::make_shared<FloatArrayHashPartitioner>(pool);
    case arrow::Type::DOUBLE:return std::make_shared<DoubleArrayHashPartitioner>(pool);
    case arrow::Type::STRING:return std::make_shared<StringHashPartitioner>(pool);
    case arrow::Type::BINARY:return std::make_shared<BinaryHashPartitionKernel>(pool);
    case arrow::Type::FIXED_SIZE_BINARY:return std::make_shared<FixedSizeBinaryHashPartitionKernel>(pool);
    default: return NULLPTR;
  }
}

std::shared_ptr<ArrowPartitionKernel> GetPartitionKernel(arrow::MemoryPool *pool,
                                                     const std::shared_ptr<arrow::Array> &values) {
  return GetPartitionKernel(pool, values->type());
}

cylon::Status HashPartitionArray(arrow::MemoryPool *pool,
                                 const std::shared_ptr<arrow::Array> &values,
                                 const std::vector<int> &targets,
                                 std::vector<int64_t> *outPartitions,
                                 std::vector<uint32_t> &counts) {
  std::shared_ptr<ArrowPartitionKernel> kernel = GetPartitionKernel(pool, values);
  if (kernel->Partition(values, targets, outPartitions, counts) == 0){
    return cylon::Status::OK();
  } else {
    return cylon::Status(cylon::Code::ExecutionError, "Hash partition returned non-zero!");
  }
}

cylon::Status HashPartitionArrays(arrow::MemoryPool *pool,
                                  const std::vector<std::shared_ptr<arrow::Array>> &values,
                                  int64_t length,
                                  const std::vector<int> &targets,
                                  std::vector<int64_t> *outPartitions,
                                  std::vector<uint32_t> &counts) {
  std::vector<std::shared_ptr<ArrowPartitionKernel>> hash_kernels;
  for (const auto &array : values) {
    auto hash_kernel = GetPartitionKernel(pool, array);
    if (hash_kernel == NULLPTR) {
      LOG(FATAL) << "Un-known type";
      return cylon::Status(cylon::NotImplemented, "Not implemented or unsupported data type.");
    }
    hash_kernels.push_back(hash_kernel);
  }

  for (int64_t index = 0; index < length; index++) {
    int64_t hash_code = 1;
    int64_t array_index = 0;
    for (const auto &array : values) {
      hash_code = 31 * hash_code + hash_kernels[array_index++]->ToHash(array, index);
    }
    int kX = targets[hash_code % targets.size()];
    outPartitions->push_back(kX);
    counts[kX]++;
  }
  return cylon::Status::OK();
}

template<template<typename T> class PART_KERNEL>
static Status create_partition_kernel(const std::shared_ptr<arrow::DataType> &data_type,
                                      uint32_t num_partitions,
                                      std::unique_ptr<ArrowPartitionKernel2> &kern) {
  switch (data_type->id()) {
    case arrow::Type::BOOL:kern = std::make_unique<PART_KERNEL<arrow::BooleanType>>(num_partitions);
      break;
    case arrow::Type::UINT8:kern = std::make_unique<PART_KERNEL<arrow::UInt8Type>>(num_partitions);
      break;
    case arrow::Type::INT8:kern = std::make_unique<PART_KERNEL<arrow::Int8Type>>(num_partitions);
      break;
    case arrow::Type::UINT16:
      kern = std::make_unique<PART_KERNEL<arrow::UInt16Type>>(num_partitions);
      break;
    case arrow::Type::INT16:kern = std::make_unique<PART_KERNEL<arrow::Int16Type>>(num_partitions);
      break;
    case arrow::Type::UINT32:
      kern = std::make_unique<PART_KERNEL<arrow::UInt32Type>>(num_partitions);
      break;
    case arrow::Type::INT32:kern = std::make_unique<PART_KERNEL<arrow::Int32Type>>(num_partitions);
      break;
    case arrow::Type::UINT64:
      kern = std::make_unique<PART_KERNEL<arrow::UInt64Type>>(num_partitions);
      break;
    case arrow::Type::INT64:kern = std::make_unique<PART_KERNEL<arrow::Int64Type>>(num_partitions);
      break;
    case arrow::Type::FLOAT:kern = std::make_unique<PART_KERNEL<arrow::FloatType>>(num_partitions);
      break;
    case arrow::Type::DOUBLE:
      kern = std::make_unique<PART_KERNEL<arrow::DoubleType>>(num_partitions);
      break;
    default:LOG_AND_RETURN_ERROR(Code::Invalid, "modulo partition works only for integer values")
  } 
  return Status::OK();
};

Status CreateHashPartitionKernel(const std::shared_ptr<arrow::DataType> &data_type,
                                 uint32_t num_partitions,
                                 std::unique_ptr<ArrowPartitionKernel2> &kern) {
//  switch (data_type->id()) {
//    case arrow::Type::BOOL:
//      kern = std::make_unique<HashPartitionKernel<arrow::BooleanType>>(num_partitions);
//      break;
//    case arrow::Type::UINT8:
//      kern = std::make_unique<HashPartitionKernel<arrow::UInt8Type>>(num_partitions);
//      break;
//    case arrow::Type::INT8:
//      kern = std::make_unique<HashPartitionKernel<arrow::Int8Type>>(num_partitions);
//      break;
//    case arrow::Type::UINT16:
//      kern = std::make_unique<HashPartitionKernel<arrow::UInt16Type>>(num_partitions);
//      break;
//    case arrow::Type::INT16:
//      kern = std::make_unique<HashPartitionKernel<arrow::Int16Type>>(num_partitions);
//      break;
//    case arrow::Type::UINT32:
//      kern = std::make_unique<HashPartitionKernel<arrow::UInt32Type>>(num_partitions);
//      break;
//    case arrow::Type::INT32:
//      kern = std::make_unique<HashPartitionKernel<arrow::Int32Type>>(num_partitions);
//      break;
//    case arrow::Type::UINT64:
//      kern = std::make_unique<HashPartitionKernel<arrow::UInt64Type>>(num_partitions);
//      break;
//    case arrow::Type::INT64:
//      kern = std::make_unique<HashPartitionKernel<arrow::Int64Type>>(num_partitions);
//      break;
//    case arrow::Type::FLOAT:
//      kern = std::make_unique<HashPartitionKernel<arrow::FloatType>>(num_partitions);
//      break;
//    case arrow::Type::DOUBLE:
//      kern = std::make_unique<HashPartitionKernel<arrow::DoubleType>>(num_partitions);
//      break;
//    default:LOG_AND_RETURN_ERROR(Code::Invalid, "modulo partition works only for integer values")
//  }
  return create_partition_kernel<HashPartitionKernel>(data_type, num_partitions, kern);
//  return Status::OK();
}

RowHashingKernel::RowHashingKernel(const std::vector<std::shared_ptr<arrow::Field>> &fields,
                                   arrow::MemoryPool *memory_pool) {
  for (auto const &field : fields) {
    this->hash_kernels.push_back(std::shared_ptr<ArrowPartitionKernel>(
        GetPartitionKernel(memory_pool, field->type())));
  }
}

int32_t RowHashingKernel::Hash(const std::shared_ptr<arrow::Table> &table, int64_t row) {
  int64_t hash_code = 1;
  for (int c = 0; c < table->num_columns(); ++c) {
    hash_code = 31 * hash_code + this->hash_kernels[c]->ToHash(table->column(c)->chunk(0), row);
  }
  return hash_code;
}
}  // namespace cylon

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
    case arrow::Type::UINT8:kernel = std::make_shared<UInt8ArrayHashPartitioner>(pool);
      break;
    case arrow::Type::INT8:kernel = std::make_shared<Int8ArrayHashPartitioner>(pool);
      break;
    case arrow::Type::UINT16:kernel = std::make_shared<UInt16ArrayHashPartitioner>(pool);
      break;
    case arrow::Type::INT16:kernel = std::make_shared<Int16ArrayHashPartitioner>(pool);
      break;
    case arrow::Type::UINT32:kernel = std::make_shared<UInt32ArrayHashPartitioner>(pool);
      break;
    case arrow::Type::INT32:kernel = std::make_shared<Int32ArrayHashPartitioner>(pool);
      break;
    case arrow::Type::UINT64:kernel = std::make_shared<UInt64ArrayHashPartitioner>(pool);
      break;
    case arrow::Type::INT64:kernel = std::make_shared<Int64ArrayHashPartitioner>(pool);
      break;
    case arrow::Type::FLOAT:kernel = std::make_shared<FloatArrayHashPartitioner>(pool);
      break;
    case arrow::Type::DOUBLE:kernel = std::make_shared<DoubleArrayHashPartitioner>(pool);
      break;
    case arrow::Type::STRING:kernel = std::make_shared<StringHashPartitioner>(pool);
      break;
    case arrow::Type::BINARY:kernel = std::make_shared<BinaryHashPartitionKernel>(pool);
      break;
    default:LOG(FATAL) << "Un-known type";
      return NULLPTR;
  }
  return kernel;
}

std::shared_ptr<ArrowPartitionKernel> GetPartitionKernel(arrow::MemoryPool *pool,
                                                     const std::shared_ptr<arrow::Array> &values) {
  return GetPartitionKernel(pool, values->type());
}

cylon::Status HashPartitionArray(arrow::MemoryPool *pool,
                                 const std::shared_ptr<arrow::Array> &values,
                                 const std::vector<int> &targets,
                                 std::vector<int64_t> *outPartitions) {
  std::shared_ptr<ArrowPartitionKernel> kernel = GetPartitionKernel(pool, values);
  kernel->Partition(values, targets, outPartitions);
  return cylon::Status::OK();
}

cylon::Status HashPartitionArrays(arrow::MemoryPool *pool,
                                  const std::vector<std::shared_ptr<arrow::Array>> &values,
                                  int64_t length,
                                  const std::vector<int> &targets,
                                  std::vector<int64_t> *outPartitions) {
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
    outPartitions->push_back(targets[hash_code % targets.size()]);
  }
  return cylon::Status::OK();
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

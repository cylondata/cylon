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

namespace twisterx {

ArrowPartitionKernel *GetPartitionKernel(arrow::MemoryPool *pool,
                                         const std::shared_ptr<arrow::DataType> &data_type) {
  ArrowPartitionKernel *kernel;
  switch (data_type->id()) {
    case arrow::Type::UINT8:kernel = new UInt8ArrayHashPartitioner(pool);
      break;
    case arrow::Type::INT8:kernel = new Int8ArrayHashPartitioner(pool);
      break;
    case arrow::Type::UINT16:kernel = new UInt16ArrayHashPartitioner(pool);
      break;
    case arrow::Type::INT16:kernel = new Int16ArrayHashPartitioner(pool);
      break;
    case arrow::Type::UINT32:kernel = new UInt32ArrayHashPartitioner(pool);
      break;
    case arrow::Type::INT32:kernel = new Int32ArrayHashPartitioner(pool);
      break;
    case arrow::Type::UINT64:kernel = new UInt64ArrayHashPartitioner(pool);
      break;
    case arrow::Type::INT64:kernel = new Int64ArrayHashPartitioner(pool);
      break;
    case arrow::Type::FLOAT:kernel = new FloatArrayHashPartitioner(pool);
      break;
    case arrow::Type::DOUBLE:kernel = new DoubleArrayHashPartitioner(pool);
      break;
    default:LOG(FATAL) << "Un-known type";
      return NULLPTR;
  }
  return kernel;
}

ArrowPartitionKernel *GetPartitionKernel(arrow::MemoryPool *pool,
                                         std::shared_ptr<arrow::Array> values) {
  return GetPartitionKernel(pool, values->type());
}

twisterx::Status HashPartitionArray(arrow::MemoryPool *pool,
                                    std::shared_ptr<arrow::Array> values,
                                    const std::vector<int> &targets,
                                    std::vector<int64_t> *outPartitions) {
  ArrowPartitionKernel *kernel = GetPartitionKernel(pool, values);
  kernel->Partition(values, targets, outPartitions);
  return twisterx::Status::OK();
}

twisterx::Status HashPartitionArrays(arrow::MemoryPool *pool,
                                     const std::vector<std::shared_ptr<arrow::Array>> &values,
                                     int64_t length,
                                     const std::vector<int> &targets,
                                     std::vector<int64_t> *outPartitions) {
  std::vector<ArrowPartitionKernel *> hash_kernels;
  for (const auto &array: values) {
    auto hash_kernel = GetPartitionKernel(pool, array);
    if (hash_kernel == NULLPTR) {
      LOG(FATAL) << "Un-known type";
      return twisterx::Status(twisterx::NotImplemented, "Not implemented or unsupported data type.");
    }
    hash_kernels.push_back(hash_kernel);
  }

  for (int64_t index = 0; index < length; index++) {
    int64_t hash_code = 1;
    int64_t array_index = 0;
    for (const auto &array: values) {
      hash_code = 31 * hash_code + hash_kernels[array_index++]->ToHash(array, index);
    }
    outPartitions->push_back(targets[hash_code % targets.size()]);
  }
  return twisterx::Status::OK();
}

RowHashingKernel::RowHashingKernel(const std::vector<std::shared_ptr<arrow::Field>> &fields,
                                   arrow::MemoryPool *memory_pool) {
  for (auto const &field: fields) {
    this->hash_kernels.push_back(std::shared_ptr<ArrowPartitionKernel>(GetPartitionKernel(memory_pool, field->type())));
  }
}

int32_t RowHashingKernel::Hash(const std::shared_ptr<arrow::Table> &table, int64_t row) {
  int64_t hash_code = 1;
  for (int c = 0; c < table->num_columns(); ++c) {
    hash_code = 31 * hash_code + this->hash_kernels[c]->ToHash(table->column(c)->chunk(0), row);
  }
  return 0;
}
}

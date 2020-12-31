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

std::unique_ptr<HashPartitionKernel> CreateHashPartitionKernel(const std::shared_ptr<arrow::DataType> &data_type) {
  switch (data_type->id()) {
    case arrow::Type::BOOL:return std::make_unique<ModuloPartitionKernel<arrow::BooleanType>>();
    case arrow::Type::UINT8:return std::make_unique<ModuloPartitionKernel<arrow::UInt8Type>>();
    case arrow::Type::INT8:return std::make_unique<ModuloPartitionKernel<arrow::Int8Type>>();
    case arrow::Type::UINT16:return std::make_unique<ModuloPartitionKernel<arrow::UInt16Type>>();
    case arrow::Type::INT16:return std::make_unique<ModuloPartitionKernel<arrow::Int16Type>>();
    case arrow::Type::UINT32:return std::make_unique<ModuloPartitionKernel<arrow::UInt32Type>>();
    case arrow::Type::INT32:return std::make_unique<ModuloPartitionKernel<arrow::Int32Type>>();
    case arrow::Type::UINT64:return std::make_unique<ModuloPartitionKernel<arrow::UInt64Type>>();
    case arrow::Type::INT64:return std::make_unique<ModuloPartitionKernel<arrow::Int64Type>>();
    case arrow::Type::FLOAT:return std::make_unique<NumericHashPartitionKernel<arrow::FloatType>>();
    case arrow::Type::DOUBLE:return std::make_unique<NumericHashPartitionKernel<arrow::DoubleType>>();
    case arrow::Type::STRING:return std::make_unique<StringHashPartitioner>();
    case arrow::Type::BINARY:return std::make_unique<BinaryHashPartitionKernel>();
    case arrow::Type::FIXED_SIZE_BINARY:return std::make_unique<FixedSizeBinaryHashPartitionKernel>();
    default: return nullptr;
  }
}

std::unique_ptr<PartitionKernel> CreateRangePartitionKernel(const std::shared_ptr<arrow::DataType> &data_type,
                                                            std::shared_ptr<CylonContext> &ctx,
                                                            bool ascending,
                                                            uint64_t num_samples,
                                                            uint32_t num_bins) {
  switch (data_type->id()) {
    case arrow::Type::BOOL:
      return std::make_unique<RangePartitionKernel<arrow::BooleanType>>(ctx,
                                                                        ascending,
                                                                        num_samples,
                                                                        num_bins);
    case arrow::Type::UINT8:
      return std::make_unique<RangePartitionKernel<arrow::UInt8Type>>(ctx,
                                                                      ascending,
                                                                      num_samples,
                                                                      num_bins);
    case arrow::Type::INT8:
      return std::make_unique<RangePartitionKernel<arrow::Int8Type>>(ctx,
                                                                     ascending,
                                                                     num_samples,
                                                                     num_bins);
    case arrow::Type::UINT16:
      return std::make_unique<RangePartitionKernel<arrow::UInt16Type>>(ctx,
                                                                       ascending,
                                                                       num_samples,
                                                                       num_bins);
    case arrow::Type::INT16:
      return std::make_unique<RangePartitionKernel<arrow::Int16Type>>(ctx,
                                                                      ascending,
                                                                      num_samples,
                                                                      num_bins);
    case arrow::Type::UINT32:
      return std::make_unique<RangePartitionKernel<arrow::UInt32Type>>(ctx,
                                                                       ascending,
                                                                       num_samples,
                                                                       num_bins);
    case arrow::Type::INT32:
      return std::make_unique<RangePartitionKernel<arrow::Int32Type>>(ctx,
                                                                      ascending,
                                                                      num_samples,
                                                                      num_bins);
    case arrow::Type::UINT64:
      return std::make_unique<RangePartitionKernel<arrow::UInt64Type>>(ctx,
                                                                       ascending,
                                                                       num_samples,
                                                                       num_bins);
    case arrow::Type::INT64:
      return std::make_unique<RangePartitionKernel<arrow::Int64Type>>(ctx,
                                                                      ascending,
                                                                      num_samples,
                                                                      num_bins);
    case arrow::Type::FLOAT:
      return std::make_unique<RangePartitionKernel<arrow::FloatType>>(ctx,
                                                                      ascending,
                                                                      num_samples,
                                                                      num_bins);
    case arrow::Type::DOUBLE:
      return std::make_unique<RangePartitionKernel<arrow::DoubleType>>(ctx,
                                                                       ascending,
                                                                       num_samples,
                                                                       num_bins);
    default:return nullptr;
  }
}

RowHashingKernel::RowHashingKernel(const std::vector<std::shared_ptr<arrow::Field>> &fields) {
  for (auto const &field : fields) {
    this->hash_kernels.push_back(CreateHashPartitionKernel(field->type()));
  }
}

int32_t RowHashingKernel::Hash(const std::shared_ptr<arrow::Table> &table, int64_t row) {
  int64_t hash_code = 1;
  for (int c = 0; c < table->num_columns(); ++c) {
    hash_code = 31 * hash_code + this->hash_kernels[c]->ToHash(table->column(c)->chunk(0), row);
  }
  return hash_code;
}

PartialRowHashingKernel::PartialRowHashingKernel(const std::vector<std::shared_ptr<arrow::Field>> &fields,
                                                 const std::vector<int> &cols) {
  for (auto const &field : fields) {
    this->hash_kernels.push_back(CreateHashPartitionKernel(field->type()));
  }
  this->columns = cols;
}

int32_t PartialRowHashingKernel::Hash(const std::shared_ptr<arrow::Table> &table,
                                      int64_t row) {
  int64_t hash_code = 1;
  for (size_t c = 0; c < columns.size(); ++c) {
    hash_code = 31 * hash_code + this->hash_kernels[c]->ToHash(table->column(c)->chunk(0), row);
  }
  return hash_code;
}

}  // namespace cylon

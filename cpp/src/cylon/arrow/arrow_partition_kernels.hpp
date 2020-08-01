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

#ifndef CYLON_ARROW_PARTITION_KERNELS_H
#define CYLON_ARROW_PARTITION_KERNELS_H

#include <memory>
#include <vector>
#include <arrow/api.h>
#include <glog/logging.h>

#include "../util/murmur3.hpp"
#include "../status.hpp"

namespace cylon {

class ArrowPartitionKernel {
 public:
  explicit ArrowPartitionKernel(
      arrow::MemoryPool *pool) : pool_(pool) {}

  /**
   * We partition the table and return the indexes as an array
   * @param ctx
   * @param values
   * @param out
   * @return
   */
  virtual int Partition(const std::shared_ptr<arrow::Array> &values,
                        const std::vector<int> &targets,
                        std::vector<int64_t> *partitions) = 0;

  virtual uint32_t ToHash(const std::shared_ptr<arrow::Array> &values,
                          int64_t index) = 0;
 protected:
  arrow::MemoryPool *pool_;
};

class FixedSizeBinaryHashPartitionKernel : public ArrowPartitionKernel {
 public:
  explicit FixedSizeBinaryHashPartitionKernel(arrow::MemoryPool *pool) :
           ArrowPartitionKernel(pool) {}

  uint32_t ToHash(const std::shared_ptr<arrow::Array> &values, int64_t index) override {
    auto reader = std::static_pointer_cast<arrow::FixedSizeBinaryArray>(values);
    if (values->IsNull(index)) {
      return 0;
    } else {
      auto val = reader->GetValue(index);
      uint32_t hash = 0;
      uint32_t seed = 0;
      // do the hash as we know the bit width
      cylon::util::MurmurHash3_x86_32(val, reader->byte_width(), seed, &hash);
      return hash;
    }
  }

  int Partition(const std::shared_ptr<arrow::Array> &values, const std::vector<int> &targets,
                std::vector<int64_t> *partitions) override {
    auto reader = std::static_pointer_cast<arrow::FixedSizeBinaryArray>(values);
    for (int64_t i = 0; i < reader->length(); i++) {
      auto lValue = reader->GetValue(i);
      uint32_t hash = 0;
      uint32_t seed = 0;
      // do the hash as we know the bit width
      cylon::util::MurmurHash3_x86_32(lValue, reader->byte_width(), seed, &hash);
      partitions->push_back(targets.at(hash % targets.size()));
    }
    // now build the
    return 0;
  }
};

class BinaryHashPartitionKernel : public ArrowPartitionKernel {
 public:
  explicit BinaryHashPartitionKernel(arrow::MemoryPool *pool) : ArrowPartitionKernel(pool) {}

  uint32_t ToHash(const std::shared_ptr<arrow::Array> &values, int64_t index) override {
    auto reader = std::static_pointer_cast<arrow::BinaryArray>(values);
    if (values->IsNull(index)) {
      return 0;
    } else {
      int length = 0;
      auto val = reader->GetValue(index, &length);
      uint32_t hash = 0;
      uint32_t seed = 0;
      // do the hash as we know the bit width
      cylon::util::MurmurHash3_x86_32(val, length, seed, &hash);
      return hash;
    }
  }

  int Partition(const std::shared_ptr<arrow::Array> &values, const std::vector<int> &targets,
                std::vector<int64_t> *partitions) override {
    auto reader = std::static_pointer_cast<arrow::BinaryArray>(values);
    for (int64_t i = 0; i < reader->length(); i++) {
      int length = 0;
      auto lValue = reader->GetValue(i, &length);
      uint32_t hash = 0;
      uint32_t seed = 0;
      // do the hash as we know the bit width
      cylon::util::MurmurHash3_x86_32(lValue, length, seed, &hash);
      partitions->push_back(targets.at(hash % targets.size()));
    }
    // now build the
    return 0;
  }
};

template<typename TYPE, typename CTYPE>
class NumericHashPartitionKernel : public ArrowPartitionKernel {
 public:
  explicit NumericHashPartitionKernel(arrow::MemoryPool *pool) : ArrowPartitionKernel(pool) {}

  uint32_t ToHash(const std::shared_ptr<arrow::Array> &values,
                  int64_t index) override {
    auto reader = std::static_pointer_cast<arrow::NumericArray<TYPE>>(values);
    auto type = std::static_pointer_cast<arrow::FixedWidthType>(values->type());
    int bitWidth = type->bit_width();
    if (values->IsNull(index)) {
      return 0;
    } else {
      CTYPE lValue = reader->Value(index);

      uint32_t hash = 0;
      uint32_t seed = 0;
      void *val = (void *) &(lValue);
      // do the hash as we know the bit width
      cylon::util::MurmurHash3_x86_32(val, bitWidth / 8, seed, &hash);
      return hash;
    }

  }

  int Partition(const std::shared_ptr<arrow::Array> &values, const std::vector<int> &targets,
                std::vector<int64_t> *partitions) override {
    auto reader = std::static_pointer_cast<arrow::NumericArray<TYPE>>(values);
    auto type = std::static_pointer_cast<arrow::FixedWidthType>(values->type());
    int bitWidth = type->bit_width();
    for (int64_t i = 0; i < reader->length(); i++) {
      auto lValue = reader->Value(i);
      void *val = (void *) &(lValue);
      uint32_t hash = 0;
      uint32_t seed = 0;
      // do the hash as we know the bit width
      cylon::util::MurmurHash3_x86_32(val, bitWidth / 8, seed, &hash);
      partitions->push_back(targets.at(hash % targets.size()));
    }
    // now build the
    return 0;
  }
};

using UInt8ArrayHashPartitioner = NumericHashPartitionKernel<arrow::UInt8Type, uint8_t>;
using UInt16ArrayHashPartitioner = NumericHashPartitionKernel<arrow::UInt16Type, uint16_t>;
using UInt32ArrayHashPartitioner = NumericHashPartitionKernel<arrow::UInt32Type, uint32_t>;
using UInt64ArrayHashPartitioner = NumericHashPartitionKernel<arrow::UInt64Type, uint64_t>;
using Int8ArrayHashPartitioner = NumericHashPartitionKernel<arrow::Int8Type, int8_t>;
using Int16ArrayHashPartitioner = NumericHashPartitionKernel<arrow::Int16Type, int16_t>;
using Int32ArrayHashPartitioner = NumericHashPartitionKernel<arrow::Int32Type, int32_t>;
using Int64ArrayHashPartitioner = NumericHashPartitionKernel<arrow::Int64Type, int64_t>;
using HalfFloatArrayHashPartitioner = NumericHashPartitionKernel<arrow::HalfFloatType, float_t>;
using FloatArrayHashPartitioner = NumericHashPartitionKernel<arrow::FloatType, float_t>;
using DoubleArrayHashPartitioner = NumericHashPartitionKernel<arrow::DoubleType, double_t>;
using StringHashPartitioner = BinaryHashPartitionKernel;
using BinaryHashPartitioner = BinaryHashPartitionKernel;

std::shared_ptr<ArrowPartitionKernel> GetPartitionKernel(arrow::MemoryPool *pool,
                                 const std::shared_ptr<arrow::Array> &values);

std::shared_ptr<ArrowPartitionKernel> GetPartitionKernel(arrow::MemoryPool *pool,
                                 const std::shared_ptr<arrow::DataType> &data_type);

cylon::Status HashPartitionArray(arrow::MemoryPool *pool,
                                 const std::shared_ptr<arrow::Array> &values,
                                 const std::vector<int> &targets,
                                 std::vector<int64_t> *outPartitions);

cylon::Status HashPartitionArrays(arrow::MemoryPool *pool,
                                  const std::vector<std::shared_ptr<arrow::Array>> &values,
                                  int64_t length,
                                  const std::vector<int> &targets,
                                  std::vector<int64_t> *outPartitions);

class RowHashingKernel {
 private:
  std::vector<std::shared_ptr<ArrowPartitionKernel>> hash_kernels;
 public:
  RowHashingKernel(const std::vector<std::shared_ptr<arrow::Field>> &vector,
                   arrow::MemoryPool *memory_pool);
  int32_t Hash(const std::shared_ptr<arrow::Table> &table, int64_t row);
};
}  // namespace cylon

#endif //CYLON_ARROW_PARTITION_KERNELS_H

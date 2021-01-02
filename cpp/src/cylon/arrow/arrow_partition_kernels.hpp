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
#include <cmath>
#include <util/macros.hpp>
#include <ctx/cylon_context.hpp>

#include "../data_types.hpp"
#include "../util/murmur3.hpp"
#include "../util/arrow_utils.hpp"
#include "../status.hpp"
#include "../arrow/arrow_types.hpp"
//#include "../compute/aggregates.hpp"
#include "../net/mpi/mpi_operations.hpp"

namespace cylon {

template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
static inline constexpr bool if_power2(T v) {
  return v && !(v & (v - 1));
}

template<typename ARROW_ARRAY_T>
using LoopRunner = std::function<void(const std::shared_ptr<ARROW_ARRAY_T> &casted_arr,
                                      uint64_t global_idx,
                                      uint64_t idx)>;

template<typename ARROW_ARRAY_T>
static inline void run_loop(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                            const LoopRunner<ARROW_ARRAY_T> &runner) {
  uint64_t offset = 0;
  for (const auto &arr: idx_col->chunks()) {
    const std::shared_ptr<ARROW_ARRAY_T> &carr = std::static_pointer_cast<ARROW_ARRAY_T>(arr);
    for (int64_t i = 0; i < carr->length(); i++, offset++) {
      runner(carr, offset, i);
    }
  }
}

using Partitioner = std::function<uint32_t(uint64_t)>;

static inline Partitioner get_partitioner(uint32_t num_partitions) {
  if (if_power2(num_partitions)) {
    return [num_partitions](uint32_t val) {
      return val & (num_partitions - 1);
    };
  } else {
    return [num_partitions](uint32_t val) {
      return val % num_partitions;
    };
  }
}


class PartitionKernel {
 public:
  virtual ~PartitionKernel() = default;

  Status Partition(const std::shared_ptr<arrow::Array> &idx_col,
                   uint32_t num_partitions,
                   std::vector<uint32_t> &target_partitions,
                   std::vector<uint32_t> &partition_histogram) {
    return Partition(std::make_shared<arrow::ChunkedArray>(idx_col),
                     num_partitions,
                     target_partitions,
                     partition_histogram);
  }

  virtual Status Partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                           uint32_t num_partitions,
                           std::vector<uint32_t> &target_partitions,
                           std::vector<uint32_t> &partition_histogram) = 0;
};

class HashPartitionKernel : public PartitionKernel {
 public:
  Status UpdateHash(const std::shared_ptr<arrow::Array> &idx_col, std::vector<uint32_t> &partial_hashes) {
    return UpdateHash(std::make_shared<arrow::ChunkedArray>(idx_col), partial_hashes);
  }

  virtual Status UpdateHash(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                            std::vector<uint32_t> &partial_hashes) = 0;

  virtual uint32_t ToHash(const std::shared_ptr<arrow::Array> &values, int64_t index) = 0;
};


template<typename ARROW_T, typename = typename std::enable_if<
    arrow::is_integer_type<ARROW_T>::value | arrow::is_boolean_type<ARROW_T>::value>::type>
class ModuloPartitionKernel : public HashPartitionKernel {
  using ARROW_ARRAY_T = typename arrow::TypeTraits<ARROW_T>::ArrayType;
  using ARROW_CTYPE = typename arrow::TypeTraits<ARROW_T>::CType;

 public:
  Status Partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                   uint32_t num_partitions,
                   std::vector<uint32_t> &target_partitions,
                   std::vector<uint32_t> &partition_histogram) override {
    if (partition_histogram.size() != num_partitions
        || target_partitions.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "target partitions or histogram not initialized!");
    }

    Partitioner partitioner = get_partitioner(num_partitions);
    LoopRunner<ARROW_ARRAY_T> loop_runner = [&](const std::shared_ptr<ARROW_ARRAY_T> &casted_arr,
                                                uint64_t chunk_offset,
                                                uint64_t idx) {
      uint32_t pseudo_hash = 31 * target_partitions[chunk_offset] + static_cast<uint32_t>(casted_arr->Value(idx));
      uint32_t p = partitioner(pseudo_hash);
      target_partitions[chunk_offset] = p;
      partition_histogram[p]++;
    };

    run_loop(idx_col, loop_runner);

    return Status::OK();
  }

  Status UpdateHash(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                    std::vector<uint32_t> &partial_hashes) override {
    if (partial_hashes.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "partial hashes size != idx col length!");
    }

    LoopRunner<ARROW_ARRAY_T> loop_runner = [&](const std::shared_ptr<ARROW_ARRAY_T> &carr,
                                                uint64_t offset,
                                                uint64_t i) {
      partial_hashes[offset] = 31 * partial_hashes[offset] + static_cast<uint32_t>(carr->Value(i));
    };
    run_loop(idx_col, loop_runner);

    return Status::OK();
  }

  inline uint32_t ToHash(const std::shared_ptr<arrow::Array> &values, int64_t index) override {
    const std::shared_ptr<ARROW_ARRAY_T> &carr = std::static_pointer_cast<ARROW_ARRAY_T>(values);
    return static_cast<uint32_t>(carr->Value(index));
  }
};


template<typename ARROW_T, typename = typename std::enable_if<
    arrow::is_number_type<ARROW_T>::value | arrow::is_boolean_type<ARROW_T>::value>::type>
class NumericHashPartitionKernel : public HashPartitionKernel {
  using ARROW_ARRAY_T = typename arrow::TypeTraits<ARROW_T>::ArrayType;
  using ARROW_CTYPE = typename arrow::TypeTraits<ARROW_T>::CType;

 public:
  Status Partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                   uint32_t num_partitions,
                   std::vector<uint32_t> &target_partitions,
                   std::vector<uint32_t> &partition_histogram) override {
    if (partition_histogram.size() != num_partitions
        || target_partitions.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "target partitions or histogram not initialized!");
    }

    Partitioner partitioner = get_partitioner(num_partitions);
    const int len = sizeof(ARROW_CTYPE);
    LoopRunner<ARROW_ARRAY_T> loop_runner = [&](const std::shared_ptr<ARROW_ARRAY_T> &carr,
                                                uint64_t global_idx,
                                                uint64_t idx) {
      ARROW_CTYPE val = carr->Value(idx);
      uint32_t hash = 0;
      util::MurmurHash3_x86_32(&val, len, 0, &hash);
      hash += 31 * target_partitions[global_idx];
      uint32_t p = partitioner(hash);
      target_partitions[global_idx] = p;
      partition_histogram[p]++;
    };
    run_loop(idx_col, loop_runner);

    return Status::OK();
  }

  Status UpdateHash(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                    std::vector<uint32_t> &partial_hashes) override {
    if (partial_hashes.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "partial hashes size != idx col length!");
    }
    const int len = sizeof(ARROW_CTYPE);
    LoopRunner<ARROW_ARRAY_T> loop_runner = [&](const std::shared_ptr<ARROW_ARRAY_T> &carr,
                                                uint64_t global_idx,
                                                uint64_t idx) {
      ARROW_CTYPE val = carr->Value(idx);
      uint32_t hash = 0;
      util::MurmurHash3_x86_32(&val, len, 0, &hash);
      hash += 31 * partial_hashes[global_idx];
      partial_hashes[global_idx] = hash;
    };
    run_loop(idx_col, loop_runner);
    return Status::OK();
  }

  uint32_t ToHash(const std::shared_ptr<arrow::Array> &values, int64_t index) override {
    if (values->IsNull(index)) {
      return 0;
    } else {
      auto reader = std::static_pointer_cast<ARROW_ARRAY_T>(values);
      ARROW_CTYPE lValue = reader->Value(index);
      uint32_t hash = 0;
      cylon::util::MurmurHash3_x86_32(&lValue, sizeof(ARROW_CTYPE), 0, &hash);
      return hash;
    }
  }
};

class FixedSizeBinaryHashPartitionKernel : public HashPartitionKernel {
 public:
  uint32_t ToHash(const std::shared_ptr<arrow::Array> &values, int64_t index) override {
    if (values->IsNull(index)) {
      return 0;
    } else {
      auto reader = std::static_pointer_cast<arrow::FixedSizeBinaryArray>(values);
      uint32_t hash = 0;
      cylon::util::MurmurHash3_x86_32(reader->GetValue(index), reader->byte_width(), 0, &hash);
      return hash;
    }
  }

  Status Partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                   uint32_t num_partitions,
                   std::vector<uint32_t> &target_partitions,
                   std::vector<uint32_t> &partition_histogram) override {
    if (partition_histogram.size() != num_partitions
        || target_partitions.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "target partitions or histogram not initialized!");
    }

    Partitioner partitioner = get_partitioner(num_partitions);
    int32_t byte_width = std::static_pointer_cast<arrow::FixedSizeBinaryType>(idx_col->type())->byte_width();
    LoopRunner<arrow::FixedSizeBinaryArray> loop_runner = [&](const std::shared_ptr<arrow::FixedSizeBinaryArray> &carr,
                                                              uint64_t global_idx,
                                                              uint64_t idx) {
      uint32_t hash = 0;
      util::MurmurHash3_x86_32(carr->GetValue(idx), byte_width, 0, &hash);
      hash += 31 * target_partitions[global_idx];
      uint32_t p = partitioner(hash);
      target_partitions[global_idx] = p;
      partition_histogram[p]++;
    };
    run_loop(idx_col, loop_runner);

    return Status::OK();
  }

  Status UpdateHash(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                    std::vector<uint32_t> &partial_hashes) override {
    if (partial_hashes.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "partial hashes size != idx col length!");
    }

    int32_t byte_width = std::static_pointer_cast<arrow::FixedSizeBinaryType>(idx_col->type())->byte_width();
    LoopRunner<arrow::FixedSizeBinaryArray> loop_runner = [&](const std::shared_ptr<arrow::FixedSizeBinaryArray> &carr,
                                                              uint64_t global_idx,
                                                              uint64_t idx) {
      uint32_t hash = 0;
      util::MurmurHash3_x86_32(carr->GetValue(idx), byte_width, 0, &hash);
      hash += 31 * partial_hashes[global_idx];
      partial_hashes[global_idx] = hash;
    };
    run_loop(idx_col, loop_runner);

    return Status::OK();
  }
};

class BinaryHashPartitionKernel : public HashPartitionKernel {
 public:
  uint32_t ToHash(const std::shared_ptr<arrow::Array> &values, int64_t index) override {
    if (values->IsNull(index)) {
      return 0;
    } else {
      auto reader = std::static_pointer_cast<arrow::BinaryArray>(values);
      int length = 0;
      const uint8_t *val = reader->GetValue(index, &length);
      uint32_t hash = 0;
      cylon::util::MurmurHash3_x86_32(val, length, 0, &hash);
      return hash;
    }
  }

  Status Partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                   uint32_t num_partitions,
                   std::vector<uint32_t> &target_partitions,
                   std::vector<uint32_t> &partition_histogram) override {
    if (partition_histogram.size() != num_partitions
        || target_partitions.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "target partitions or histogram not initialized!");
    }

    Partitioner partitioner = get_partitioner(num_partitions);
    LoopRunner<arrow::BinaryArray> loop_runner = [&](const std::shared_ptr<arrow::BinaryArray> &carr,
                                                     uint64_t global_idx,
                                                     uint64_t idx) {
      int32_t byte_width = 0;
      uint32_t hash = 0;
      const uint8_t *val = carr->GetValue(idx, &byte_width);
      util::MurmurHash3_x86_32(val, byte_width, 0, &hash);
      hash += 31 * target_partitions[global_idx];
      uint32_t p = partitioner(hash);
      target_partitions[global_idx] = p;
      partition_histogram[p]++;
    };
    run_loop(idx_col, loop_runner);

    return Status::OK();
  }

  Status UpdateHash(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                    std::vector<uint32_t> &partial_hashes) override {
    if (partial_hashes.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "partial hashes size != idx col length!");
    }

    LoopRunner<arrow::BinaryArray> loop_runner = [&](const std::shared_ptr<arrow::BinaryArray> &carr,
                                                     uint64_t global_idx,
                                                     uint64_t idx) {
      int32_t byte_width = 0;
      uint32_t hash = 0;
      const uint8_t *val = carr->GetValue(idx, &byte_width);
      util::MurmurHash3_x86_32(val, byte_width, 0, &hash);
      hash += 31 * partial_hashes[global_idx];
      partial_hashes[global_idx] = hash;
    };
    run_loop(idx_col, loop_runner);

    return Status::OK();
  }
};

using UInt8ArrayHashPartitioner = NumericHashPartitionKernel<arrow::UInt8Type>;
using UInt16ArrayHashPartitioner = NumericHashPartitionKernel<arrow::UInt16Type>;
using UInt32ArrayHashPartitioner = NumericHashPartitionKernel<arrow::UInt32Type>;
using UInt64ArrayHashPartitioner = NumericHashPartitionKernel<arrow::UInt64Type>;
using Int8ArrayHashPartitioner = NumericHashPartitionKernel<arrow::Int8Type>;
using Int16ArrayHashPartitioner = NumericHashPartitionKernel<arrow::Int16Type>;
using Int32ArrayHashPartitioner = NumericHashPartitionKernel<arrow::Int32Type>;
using Int64ArrayHashPartitioner = NumericHashPartitionKernel<arrow::Int64Type>;
using HalfFloatArrayHashPartitioner = NumericHashPartitionKernel<arrow::HalfFloatType>;
using FloatArrayHashPartitioner = NumericHashPartitionKernel<arrow::FloatType>;
using DoubleArrayHashPartitioner = NumericHashPartitionKernel<arrow::DoubleType>;
using StringHashPartitioner = BinaryHashPartitionKernel;
using BinaryHashPartitioner = BinaryHashPartitionKernel;

/**
 * creates hash partition kernel
 * @param data_type
 * @return
 */
std::unique_ptr<HashPartitionKernel> CreateHashPartitionKernel(const std::shared_ptr<arrow::DataType> &data_type);


class RowHashingKernel {
 private:
  std::vector<std::unique_ptr<HashPartitionKernel>> hash_kernels;
 public:
  explicit RowHashingKernel(const std::vector<std::shared_ptr<arrow::Field>> &fields);

  int32_t Hash(const std::shared_ptr<arrow::Table> &table, int64_t row);
};

class PartialRowHashingKernel {
 private:
  std::vector<std::unique_ptr<HashPartitionKernel>> hash_kernels;
  std::vector<int> columns;
 public:
  explicit PartialRowHashingKernel(const std::vector<std::shared_ptr<arrow::Field>> &fields, const std::vector<int> &cols);

  int32_t Hash(const std::shared_ptr<arrow::Table> &table, int64_t row);
};




/**
 * create range partition
 * @param data_type
 * @param num_partitions
 * @param ctx
 * @param ascending
 * @param num_samples
 * @param num_bins
 * @return
 */
std::unique_ptr<PartitionKernel> CreateRangePartitionKernel(const std::shared_ptr<arrow::DataType> &data_type,
                                                            std::shared_ptr<CylonContext> &ctx,
                                                            bool ascending,
                                                            uint64_t num_samples,
                                                            uint32_t num_bins);

}  // namespace cylon

#endif //CYLON_ARROW_PARTITION_KERNELS_H

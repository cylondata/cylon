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
#include "../compute/aggregates.hpp"
#include "../net/mpi/mpi_operations.hpp"

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
                        std::vector<int64_t> *partitions,
                        std::vector<uint32_t> &counts) = 0;

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

  int Partition(const std::shared_ptr<arrow::Array> &values,
                const std::vector<int> &targets,
                std::vector<int64_t> *partitions,
                std::vector<uint32_t> &counts) override {
    auto reader = std::static_pointer_cast<arrow::FixedSizeBinaryArray>(values);
    int64_t kI = reader->length();
    unsigned long target_size = targets.size();
    int32_t byte_width = reader->byte_width();
    for (int64_t i = 0; i < kI; i++) {
      auto lValue = reader->GetValue(i);
      uint32_t hash = 0;
      uint32_t seed = 0;
      // do the hash as we know the bit width
      cylon::util::MurmurHash3_x86_32(lValue, byte_width, seed, &hash);
      int kX = targets.at(hash % target_size);
      partitions->push_back(kX);
      counts[kX]++;
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

  int Partition(const std::shared_ptr<arrow::Array> &values,
                const std::vector<int> &targets,
                std::vector<int64_t> *partitions,
                std::vector<uint32_t> &counts) override {
    auto reader = std::static_pointer_cast<arrow::BinaryArray>(values);
    int64_t reader_len = reader->length();
    unsigned long target_size = targets.size();
    for (int64_t i = 0; i < reader_len; i++) {
      int length = 0;
      auto lValue = reader->GetValue(i, &length);
      uint32_t hash = 0;
      uint32_t seed = 0;
      // do the hash as we know the bit width
      cylon::util::MurmurHash3_x86_32(lValue, length, seed, &hash);
      int kX = targets.at(hash % target_size);
      partitions->push_back(kX);
      counts[kX]++;
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

  int Partition(const std::shared_ptr<arrow::Array> &values,
                const std::vector<int> &targets,
                std::vector<int64_t> *partitions,
                std::vector<uint32_t> &counts) override {
    auto reader = std::static_pointer_cast<arrow::NumericArray<TYPE>>(values);
    auto type = std::static_pointer_cast<arrow::FixedWidthType>(values->type());
    int bitWidth = type->bit_width() / 8;
    unsigned long target_size = targets.size();
    int64_t length = reader->length();
    for (int64_t i = 0; i < length; i++) {
      auto lValue = reader->Value(i);
      void *val = (void *) &(lValue);
      uint32_t hash = 0;
      uint32_t seed = 0;
      // do the hash as we know the bit width
      cylon::util::MurmurHash3_x86_32(val, bitWidth, seed, &hash);
      int kX = targets[hash % target_size];
      partitions->push_back(kX);
      counts[kX]++;
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
                                 std::vector<int64_t> *outPartitions,
                                 std::vector<uint32_t> &counts);

cylon::Status HashPartitionArrays(arrow::MemoryPool *pool,
                                  const std::vector<std::shared_ptr<arrow::Array>> &values,
                                  int64_t length,
                                  const std::vector<int> &targets,
                                  std::vector<int64_t> *outPartitions,
                                  std::vector<uint32_t> &counts);

class RowHashingKernel {
 private:
  std::vector<std::shared_ptr<ArrowPartitionKernel>> hash_kernels;
 public:
  RowHashingKernel(const std::vector<std::shared_ptr<arrow::Field>> &vector,
                   arrow::MemoryPool *memory_pool);
  int32_t Hash(const std::shared_ptr<arrow::Table> &table, int64_t row);
};

class ArrowPartitionKernel2 {
 public:
  explicit ArrowPartitionKernel2(uint32_t num_partitions) : num_partitions(num_partitions) {}
  virtual ~ArrowPartitionKernel2() = default;

  virtual Status Partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                           std::vector<uint32_t> &target_partitions,
                           std::vector<uint32_t> &partition_histogram) = 0;

  virtual Status UpdateHash(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                            std::vector<uint32_t> &target_partitions) = 0;

  const uint32_t num_partitions;
};

template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
static inline bool if_power2(T v) {
  return v && !(v & (v - 1));
}

struct WithDefaultPartitioner {
 public:
  virtual ~WithDefaultPartitioner() = default;

  std::function<uint32_t(uint64_t)> partitioner;

  explicit WithDefaultPartitioner(uint32_t num_partitions) {
    if (if_power2(num_partitions)) {
      partitioner = [num_partitions](uint32_t val) {
        return val & (num_partitions - 1);
      };
    } else {
      partitioner = [num_partitions](uint32_t val) {
        return val % num_partitions;
      };
    }
  };
};

template<typename ARROW_T, typename = typename std::enable_if<
    arrow::is_integer_type<ARROW_T>::value | arrow::is_boolean_type<ARROW_T>::value>::type>
class ModuloPartitionKernel : public ArrowPartitionKernel2, public WithDefaultPartitioner {
  using ARROW_ARRAY_T = typename arrow::TypeTraits<ARROW_T>::ArrayType;
  using ARROW_CTYPE = typename arrow::TypeTraits<ARROW_T>::CType;

 public:
  explicit ModuloPartitionKernel(uint32_t num_partitions)
      : ArrowPartitionKernel2(num_partitions), WithDefaultPartitioner(num_partitions) {
  }

  Status Partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                   std::vector<uint32_t> &target_partitions,
                   std::vector<uint32_t> &partition_histogram) override {
    if (partition_histogram.size() != num_partitions
        || target_partitions.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "target partitions or histogram not initialized!");
    }

    size_t offset = 0;
    for (const auto &arr: idx_col->chunks()) {
      const std::shared_ptr<ARROW_ARRAY_T> &carr = std::static_pointer_cast<ARROW_ARRAY_T>(arr);
      for (int64_t i = 0; i < carr->length(); i++, offset++) {
        // update the hash
        uint32_t hash = 31 * target_partitions[offset] + static_cast<uint32_t>(carr->Value(i));
        uint32_t p = partitioner(hash);
        target_partitions[offset] = p;
        partition_histogram[p]++;
      }
    }
    return Status::OK();
  }

  Status UpdateHash(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                    std::vector<uint32_t> &partial_hashes) override {
    if (partial_hashes.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "partial hashes size != idx col length!");
    }

    size_t offset = 0;
    for (const auto &arr: idx_col->chunks()) {
      const std::shared_ptr<ARROW_ARRAY_T> &carr = std::static_pointer_cast<ARROW_ARRAY_T>(arr);
      for (int64_t i = 0; i < carr->length(); i++, offset++) {
        // for integers, value itself can be its hash
        partial_hashes[offset] =
            31 * partial_hashes[offset] + static_cast<uint32_t>(carr->Value(i));
      }
    }
    return Status::OK();
  }
};

template<typename ARROW_T, typename = typename std::enable_if<
    arrow::is_number_type<ARROW_T>::value | arrow::is_boolean_type<ARROW_T>::value>::type>
class HashPartitionKernel : public ArrowPartitionKernel2, public WithDefaultPartitioner {
  using ARROW_ARRAY_T = typename arrow::TypeTraits<ARROW_T>::ArrayType;
  using ARROW_CTYPE = typename arrow::TypeTraits<ARROW_T>::CType;

 public:
  explicit HashPartitionKernel(uint32_t num_partitions) :
      ArrowPartitionKernel2(num_partitions), WithDefaultPartitioner(num_partitions) {
  }

  Status Partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                   std::vector<uint32_t> &target_partitions,
                   std::vector<uint32_t> &partition_histogram) override {
    if (partition_histogram.size() != num_partitions
        || target_partitions.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "target partitions or histogram not initialized!");
    }

    const int len = sizeof(ARROW_CTYPE);
    size_t offset = 0;
    for (const auto &arr: idx_col->chunks()) {
      const std::shared_ptr<ARROW_ARRAY_T> &carr = std::static_pointer_cast<ARROW_ARRAY_T>(arr);
      for (int64_t i = 0; i < carr->length(); i++, offset++) {
        ARROW_CTYPE val = carr->Value(i);
        uint32_t hash = 0;
        util::MurmurHash3_x86_32(&val, len, 0, &hash);
        hash += 31 * target_partitions[offset];
        uint32_t p = partitioner(hash);
        target_partitions[offset] = p;
        partition_histogram[p]++;
      }
    }

    return Status::OK();
  }

  Status UpdateHash(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                    std::vector<uint32_t> &partial_hashes) override {
    if (partial_hashes.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "partial hashes size != idx col length!");
    }
    const int len = sizeof(ARROW_CTYPE);
    size_t offset = 0;
    for (const auto &arr: idx_col->chunks()) {
      const std::shared_ptr<ARROW_ARRAY_T> &carr = std::static_pointer_cast<ARROW_ARRAY_T>(arr);
      for (int64_t i = 0; i < carr->length(); i++, offset++) {
        ARROW_CTYPE val = carr->Value(i);
        uint32_t hash = 0;
        util::MurmurHash3_x86_32(&val, len, 0, &hash);
        hash += 31 * partial_hashes[offset];
        partial_hashes[offset] = hash;
      }
    }
    return Status::OK();
  }
};

std::unique_ptr<ArrowPartitionKernel2> CreateHashPartitionKernel(const std::shared_ptr<arrow::DataType> &data_type,
                                                                 uint32_t num_partitions);


template<typename ARROW_T, typename = typename std::enable_if<
    arrow::is_number_type<ARROW_T>::value | arrow::is_boolean_type<ARROW_T>::value>::type>
class RangePartitionKernel : public ArrowPartitionKernel2 {
  using ARROW_ARRAY_T = typename arrow::TypeTraits<ARROW_T>::ArrayType;
  using ARROW_SCALAR_T = typename arrow::TypeTraits<ARROW_T>::ScalarType;
  using ARROW_C_T = typename arrow::TypeTraits<ARROW_T>::CType;

 public:
  RangePartitionKernel(uint32_t num_partitions,
                       std::shared_ptr<CylonContext> &ctx,
                       bool ascending,
                       uint64_t num_samples,
                       uint32_t num_bins) : ArrowPartitionKernel2(num_partitions),
                                            ascending(ascending),
                                            num_bins(num_bins),
                                            num_samples(num_samples),
                                            ctx(ctx) {
  };

  Status Partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                   std::vector<uint32_t> &target_partitions,
                   std::vector<uint32_t> &partition_histogram) override {
    if (partition_histogram.size() != num_partitions
        || target_partitions.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "target partitions or histogram not initialized!");
    }

    auto status = build_bin_to_partition(idx_col);
    RETURN_CYLON_STATUS_IF_FAILED(status)

    size_t offset = 0;
    for (const auto &chunk:idx_col->chunks()) {
      const std::shared_ptr<ARROW_ARRAY_T> &carr = std::static_pointer_cast<ARROW_ARRAY_T>(chunk);
      for (int64_t i = 0; i < carr->length(); i++, offset++) {
        ARROW_C_T val = carr->Value(i);
        uint32_t p = ascending ?
                     bin_to_partition[get_bin_pos(val)] : num_partitions - 1 - bin_to_partition[get_bin_pos(val)];
        target_partitions[offset] = p;
        partition_histogram[p]++;
      }
    }
    return Status::OK();
  }

  Status UpdateHash(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                    std::vector<uint32_t> &target_partitions) override {
    return Status(Code::Invalid, "Range partition does not build hash");
  }

 private:
  Status inline build_bin_to_partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col) {
    const std::shared_ptr<DataType> &data_type = tarrow::ToCylonType(idx_col->type());
    std::shared_ptr<arrow::ChunkedArray> sampled_chunk_array;

    if ((uint64_t) idx_col->length() == num_samples) { // if len == num_samples, dont sample, just use idx col as it is!
      sampled_chunk_array = idx_col;
    } else { // else, sample idx_col for num_samples
      std::shared_ptr<arrow::Array> sampled_array;
      auto a_status = util::SampleArray(idx_col, num_samples, sampled_array);
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(a_status)
      sampled_chunk_array = std::make_shared<arrow::ChunkedArray>(sampled_array); // create a single chunk array
    }

    // calculate minmax of the sample
    std::shared_ptr<compute::Result> minmax;
    Status status = compute::MinMax(ctx, sampled_chunk_array, data_type, minmax);
    RETURN_CYLON_STATUS_IF_FAILED(status)

    const auto &struct_scalar = minmax->GetResult().scalar_as<arrow::StructScalar>();
    min = std::static_pointer_cast<ARROW_SCALAR_T>(struct_scalar.value[0])->value;
    max = std::static_pointer_cast<ARROW_SCALAR_T>(struct_scalar.value[1])->value;
    if (arrow::is_integer_type<ARROW_T>()) {
      range_per_bin = (max - min + num_bins - 1) / num_bins; // upper bound of the division
    } else {
      range_per_bin = (max - min) / num_bins; // upper bound of the division
    }
    max = min + range_per_bin * num_bins; // update max

    // create sample histogram
    std::vector<uint64_t> local_counts(num_bins + 2, 0);
    for (const auto &chunk:sampled_chunk_array->chunks()) {
      const std::shared_ptr<ARROW_ARRAY_T> &casted_arr = std::static_pointer_cast<ARROW_ARRAY_T>(chunk);
      for (int64_t i = 0; i < casted_arr->length(); i++) {
        ARROW_C_T val = casted_arr->Value(i);
        local_counts[get_bin_pos(val)]++;
      }
    }

    sampled_chunk_array.reset();

    // all reduce local sample histograms
    std::vector<uint64_t> global_counts, *global_counts_ptr;
    if (ctx->GetWorldSize() > 1) { // if distributed, all-reduce all local bin counts
      global_counts.resize(num_bins + 2, 0);
      status = cylon::mpi::AllReduce(local_counts.data(), global_counts.data(), num_bins + 2, cylon::UInt64(),
                                     cylon::net::SUM);
      RETURN_CYLON_STATUS_IF_FAILED(status)
      global_counts_ptr = &global_counts;
      local_counts.clear();
    } else { // else, just use local bin counts
      global_counts_ptr = &local_counts;
    }

    float_t quantile = 1.0 / num_partitions, prefix_sum = 0;

    LOG(INFO) << "len=" << idx_col->length() << " min=" << min << " max=" << max << " range per bin=" <<
              range_per_bin << " num bins=" << num_bins << " quantile=" << quantile;

    // divide global histogram into quantiles
    const uint64_t total_samples = ctx->GetWorldSize() * num_samples;
    uint32_t curr_partition = 0;
    float_t target_quantile = quantile;
    for (const auto &c:*global_counts_ptr) {
      bin_to_partition.push_back(curr_partition);
      float_t freq = (float_t) c / total_samples;
      prefix_sum += freq;
      if (prefix_sum > target_quantile) {
        curr_partition += (curr_partition < num_partitions - 1); // if curr_partition < numpartition: curr_partition++
        target_quantile += quantile;
      }
    }

    if (curr_partition != num_partitions - 1) {
      LOG(WARNING) << "sample was unable to capture distribution. try increasing num_samples or num_bins";
    }
    return Status::OK();
  }

  inline constexpr size_t get_bin_pos(ARROW_C_T val) {
    return (val >= min)
        + ((val >= min) & (val < max)) * ((val - min) / range_per_bin)
        + (val >= max) * num_bins;
  }

  const bool ascending;
  const uint32_t num_bins;
  const uint64_t num_samples;
  std::shared_ptr<CylonContext> ctx; // todo dont need a copy here
  std::vector<uint32_t> bin_to_partition;
  ARROW_C_T min, max, range_per_bin;
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
std::unique_ptr<ArrowPartitionKernel2> CreateRangePartitionKernel(const std::shared_ptr<arrow::DataType> &data_type,
                                                                  uint32_t num_partitions,
                                                                  std::shared_ptr<CylonContext> &ctx,
                                                                  bool ascending,
                                                                  uint64_t num_samples,
                                                                  uint32_t num_bins);

}  // namespace cylon

#endif //CYLON_ARROW_PARTITION_KERNELS_H

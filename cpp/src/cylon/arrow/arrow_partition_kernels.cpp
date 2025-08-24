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

#include <arrow/visit_data_inline.h>

#include <cylon/util/murmur3.hpp>
#include <cylon/util/macros.hpp>
#include <cylon/util/arrow_utils.hpp>
#include <cylon/compute/aggregates.hpp>
#include <cylon/net/mpi/mpi_operations.hpp>
#include <cylon/arrow/arrow_partition_kernels.hpp>
#include <cylon/arrow/arrow_types.hpp>
#include "cylon/arrow/arrow_type_traits.hpp"

namespace cylon {

template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
static inline constexpr bool if_power2(T v) {
  return v && !(v & (v - 1));
}

template<typename T>
static inline constexpr uint32_t mod_power2(T val, uint32_t num_partitions) {
  return val & (num_partitions - 1);
}

template<typename T>
static inline constexpr uint32_t mod(T val, uint32_t num_partitions) {
  return val % num_partitions;
}

template<typename ArrowT, typename ValidFn, typename NullFn>
inline Status visit_chunked_array(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                                  ValidFn &&valid_fn,
                                  NullFn &&null_fn) {
  using ValueT = typename cylon::ArrowTypeTraits<ArrowT>::ValueT;

  uint64_t global_idx = 0;
  for (auto &&array: idx_col->chunks()) {
    const auto &arr_data = array->data();

    arrow::VisitArraySpanInline<ArrowT>(*arr_data,
                                        [&](ValueT val) {
                                          valid_fn(global_idx, val);
                                          global_idx++;
                                        },
                                        [&]() { // nothing to do for nulls
                                          null_fn(global_idx);
                                          global_idx++;
                                        }
    );
  }
  // global index should match the chunk array length
  assert(idx_col->length() == static_cast<int64_t>(global_idx));

  return Status::OK();
}

template<typename ARROW_T, typename = typename std::enable_if<
    arrow::is_integer_type<ARROW_T>::value
        | arrow::is_boolean_type<ARROW_T>::value
        | arrow::is_temporal_type<ARROW_T>::value>::type>
class ModuloPartitionKernel : public HashPartitionKernel {
  using ArrayT = typename arrow::TypeTraits<ARROW_T>::ArrayType;
  using T = typename cylon::ArrowTypeTraits<ARROW_T>::ValueT;
 public:
  Status Partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                   uint32_t num_partitions,
                   std::vector<uint32_t> &target_partitions,
                   std::vector<uint32_t> &partition_histogram) override {
    if (partition_histogram.size() != num_partitions
        || target_partitions.size() != (size_t) idx_col->length()) {
      return {Code::Invalid, "target partitions or histogram not initialized!"};
    }

    if (if_power2(num_partitions)) {
      return visit_chunked_array<ARROW_T>(
          idx_col,
          [&](uint64_t offset, T val) {
            T pseudo_hash = 31 * target_partitions[offset] + val;
            uint32_t p = mod_power2(pseudo_hash, num_partitions);
            target_partitions[offset] = p;
            partition_histogram[p]++;
          },
          [&](uint64_t offset) {
            uint32_t p = mod_power2(target_partitions[offset], num_partitions);
            target_partitions[offset] = p;
            partition_histogram[p]++;
          });
    } else {
      return visit_chunked_array<ARROW_T>(
          idx_col,
          [&](uint64_t offset, T val) {
            T pseudo_hash = 31 * target_partitions[offset] + val;
            uint32_t p = mod(pseudo_hash, num_partitions);
            target_partitions[offset] = p;
            partition_histogram[p]++;
          },
          [&](uint64_t offset) {
            uint32_t p = mod(target_partitions[offset], num_partitions);
            target_partitions[offset] = p;
            partition_histogram[p]++;
          });
    }
  }

  Status UpdateHash(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                    std::vector<uint32_t> &partial_hashes) override {
    if (partial_hashes.size() != (size_t) idx_col->length()) {
      return {Code::Invalid, "partial hashes size != idx col length!"};
    }

    return visit_chunked_array<ARROW_T>(
        idx_col,
        [&](uint64_t offset, T val) {
          partial_hashes[offset] = 31 * partial_hashes[offset] + val;
        },
        [&](uint64_t offset) {
          CYLON_UNUSED(offset);
        }); // null values wouldn't change
  }

  inline uint32_t ToHash(const std::shared_ptr<arrow::Array> &values, int64_t index) override {
    const auto &carr = std::static_pointer_cast<ArrayT>(values);
    return static_cast<uint32_t>(carr->Value(index));
  }

  static Status Make(std::unique_ptr<HashPartitionKernel> *out_kernel) {
    *out_kernel = std::make_unique<ModuloPartitionKernel<ARROW_T>>();
    return Status::OK();
  }
};

template<typename ARROW_T, typename = typename std::enable_if<
    arrow::is_number_type<ARROW_T>::value | arrow::is_boolean_type<ARROW_T>::value>::type>
class NumericHashPartitionKernel : public HashPartitionKernel {
  using ArrayT = typename arrow::TypeTraits<ARROW_T>::ArrayType;
  using ValueT = typename cylon::ArrowTypeTraits<ARROW_T>::ValueT;

 public:
  Status Partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                   uint32_t num_partitions,
                   std::vector<uint32_t> &target_partitions,
                   std::vector<uint32_t> &partition_histogram) override {
    if (partition_histogram.size() != num_partitions
        || target_partitions.size() != (size_t) idx_col->length()) {
      return {Code::Invalid, "target partitions or histogram not initialized!"};
    }

    constexpr int len = sizeof(ValueT);
    if (if_power2(num_partitions)) {
      return visit_chunked_array<ARROW_T>(idx_col,
                                          [&](uint64_t global_idx, ValueT val) {
                                            uint32_t hash = 0;
                                            util::MurmurHash3_x86_32(&val, len, 0, &hash);
                                            hash += 31 * target_partitions[global_idx];
                                            uint32_t p = mod_power2(hash, num_partitions);
                                            target_partitions[global_idx] = p;
                                            partition_histogram[p]++;
                                          },
                                          [&](uint64_t global_idx) {
                                            uint32_t p = mod_power2(target_partitions[global_idx], num_partitions);
                                            target_partitions[global_idx] = p;
                                            partition_histogram[p]++;
                                          });
    } else {
      return
          visit_chunked_array<ARROW_T>(idx_col,
                                       [&](uint64_t global_idx, ValueT val) {
                                         uint32_t hash = 0;
                                         util::MurmurHash3_x86_32(&val, len, 0, &hash);
                                         hash += 31 * target_partitions[global_idx];
                                         uint32_t p = mod(hash, num_partitions);
                                         target_partitions[global_idx] = p;
                                         partition_histogram[p]++;
                                       },
                                       [&](uint64_t global_idx) {
                                         uint32_t p = mod(target_partitions[global_idx], num_partitions);
                                         target_partitions[global_idx] = p;
                                         partition_histogram[p]++;
                                       });
    }
  }

  Status UpdateHash(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                    std::vector<uint32_t> &partial_hashes) override {
    if (partial_hashes.size() != (size_t) idx_col->length()) {
      return {Code::Invalid, "partial hashes size != idx col length!"};
    }
    constexpr int len = sizeof(ValueT);

    return visit_chunked_array<ARROW_T>(idx_col,
                                        [&](uint64_t global_idx, ValueT val) {
                                          uint32_t hash = 0;
                                          util::MurmurHash3_x86_32(&val, len, 0, &hash);
                                          hash += 31 * partial_hashes[global_idx];
                                          partial_hashes[global_idx] = hash;
                                        },
                                        [&](uint64_t global_idx) {
                                          CYLON_UNUSED(global_idx);
                                        });
  }

  uint32_t ToHash(const std::shared_ptr<arrow::Array> &values, int64_t index) override {
    if (values->IsNull(index)) {
      return 0;
    } else {
      auto reader = std::static_pointer_cast<ArrayT>(values);
      ValueT lValue = reader->Value(index);
      uint32_t hash = 0;
      cylon::util::MurmurHash3_x86_32(&lValue, sizeof(ValueT), 0, &hash);
      return hash;
    }
  }

  static Status Make(std::unique_ptr<HashPartitionKernel> *out_kernel) {
    *out_kernel = std::make_unique<NumericHashPartitionKernel<ARROW_T>>();
    return Status::OK();
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
      return {Code::Invalid, "target partitions or histogram not initialized!"};
    }

    const int32_t len = std::static_pointer_cast<arrow::FixedSizeBinaryType>(idx_col->type())->byte_width();

    if (if_power2(num_partitions)) {
      return
          visit_chunked_array<arrow::FixedSizeBinaryType>(
              idx_col,
              [&](uint64_t global_idx, std::string_view val) {
                uint32_t hash = 0;
                util::MurmurHash3_x86_32(&val, len, 0, &hash);
                hash += 31 * target_partitions[global_idx];
                uint32_t p = mod_power2(hash, num_partitions);
                target_partitions[global_idx] = p;
                partition_histogram[p]++;
              },
              [&](uint64_t global_idx) {
                uint32_t p = mod_power2(target_partitions[global_idx], num_partitions);
                target_partitions[global_idx] = p;
                partition_histogram[p]++;
              });
    } else {
      return visit_chunked_array<arrow::FixedSizeBinaryType>(
          idx_col,
          [&](uint64_t global_idx, std::string_view val) {
            uint32_t hash = 0;
            util::MurmurHash3_x86_32(&val, len, 0, &hash);
            hash += 31 * target_partitions[global_idx];
            uint32_t p = mod(hash, num_partitions);
            target_partitions[global_idx] = p;
            partition_histogram[p]++;
          },
          [&](uint64_t global_idx) {
            uint32_t p = mod(target_partitions[global_idx], num_partitions);
            target_partitions[global_idx] = p;
            partition_histogram[p]++;
          });
    }
  }

  Status UpdateHash(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                    std::vector<uint32_t> &partial_hashes) override {
    if (partial_hashes.size() != (size_t) idx_col->length()) {
      return {Code::Invalid, "partial hashes size != idx col length!"};
    }

    int32_t byte_width = std::static_pointer_cast<arrow::FixedSizeBinaryType>(idx_col->type())->byte_width();

    return visit_chunked_array<arrow::FixedSizeBinaryType>
        (idx_col,
         [&](uint64_t global_idx, std::string_view val) {
           uint32_t hash = 0;
           util::MurmurHash3_x86_32(&val, byte_width, 0, &hash);
           hash += 31 * partial_hashes[global_idx];
           partial_hashes[global_idx] = hash;
         },
         [&](uint64_t global_idx) {
           CYLON_UNUSED(global_idx);
         });
  }

  static Status Make(std::unique_ptr<HashPartitionKernel> *out_kernel) {
    *out_kernel = std::make_unique<FixedSizeBinaryHashPartitionKernel>();
    return Status::OK();
  }
};

template<typename ArrowT, typename = typename arrow::enable_if_base_binary<ArrowT>>
class BinaryHashPartitionKernel : public HashPartitionKernel {
 public:
  uint32_t ToHash(const std::shared_ptr<arrow::Array> &values, int64_t index) override {
    if (values->IsNull(index)) {
      return 0;
    } else {
      const auto &reader = std::static_pointer_cast<arrow::BinaryArray>(values);
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
      return {Code::Invalid, "target partitions or histogram not initialized!"};
    }

    if (if_power2(num_partitions)) {
      return
          visit_chunked_array<ArrowT>(
              idx_col,
              [&](uint64_t global_idx, std::string_view val) {
                uint32_t hash = 0;
                util::MurmurHash3_x86_32(&val, static_cast<int>(val.size()), 0, &hash);
                hash += 31 * target_partitions[global_idx];
                uint32_t p = mod_power2(hash, num_partitions);
                target_partitions[global_idx] = p;
                partition_histogram[p]++;
              },
              [&](uint64_t global_idx) {
                uint32_t p = mod_power2(target_partitions[global_idx], num_partitions);
                target_partitions[global_idx] = p;
                partition_histogram[p]++;
              });
    } else {
      return
          visit_chunked_array<ArrowT>(
              idx_col,
              [&](uint64_t global_idx, std::string_view val) {
                uint32_t hash = 0;
                util::MurmurHash3_x86_32(&val, static_cast<int>(val.size()), 0, &hash);
                hash += 31 * target_partitions[global_idx];
                uint32_t p = mod(hash, num_partitions);
                target_partitions[global_idx] = p;
                partition_histogram[p]++;
              },
              [&](uint64_t global_idx) {
                uint32_t p = mod(target_partitions[global_idx], num_partitions);
                target_partitions[global_idx] = p;
                partition_histogram[p]++;
              });
    }
  }

  Status UpdateHash(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                    std::vector<uint32_t> &partial_hashes) override {
    if (partial_hashes.size() != (size_t) idx_col->length()) {
      return {Code::Invalid, "partial hashes size != idx col length!"};
    }

    return
        visit_chunked_array<arrow::FixedSizeBinaryType>(
            idx_col,
            [&](uint64_t global_idx, std::string_view val) {
              uint32_t hash = 0;
              util::MurmurHash3_x86_32(&val, static_cast<int>(val.size()), 0, &hash);
              hash += 31 * partial_hashes[global_idx];
              partial_hashes[global_idx] = hash;
            },
            [&](uint64_t global_idx) {
              CYLON_UNUSED(global_idx);
            });
  }

  static Status Make(std::unique_ptr<HashPartitionKernel> *out_kernel) {
    *out_kernel = std::make_unique<BinaryHashPartitionKernel<ArrowT>>();
    return Status::OK();
  }
};

Status CreateHashPartitionKernel(const std::shared_ptr<arrow::DataType> &data_type,
                                 std::unique_ptr<HashPartitionKernel> *out_kernel) {
  switch (data_type->id()) {
    case arrow::Type::BOOL:return ModuloPartitionKernel<arrow::BooleanType>::Make(out_kernel);
    case arrow::Type::UINT8:return ModuloPartitionKernel<arrow::UInt8Type>::Make(out_kernel);
    case arrow::Type::INT8:return ModuloPartitionKernel<arrow::Int8Type>::Make(out_kernel);
    case arrow::Type::UINT16:return ModuloPartitionKernel<arrow::UInt16Type>::Make(out_kernel);
    case arrow::Type::INT16:return ModuloPartitionKernel<arrow::Int16Type>::Make(out_kernel);
    case arrow::Type::UINT32:return ModuloPartitionKernel<arrow::UInt32Type>::Make(out_kernel);
    case arrow::Type::INT32:return ModuloPartitionKernel<arrow::Int32Type>::Make(out_kernel);
    case arrow::Type::UINT64:return ModuloPartitionKernel<arrow::UInt64Type>::Make(out_kernel);
    case arrow::Type::INT64:return ModuloPartitionKernel<arrow::Int64Type>::Make(out_kernel);
    case arrow::Type::FLOAT:return NumericHashPartitionKernel<arrow::FloatType>::Make(out_kernel);
    case arrow::Type::DOUBLE:return NumericHashPartitionKernel<arrow::DoubleType>::Make(out_kernel);
    case arrow::Type::STRING: return BinaryHashPartitionKernel<arrow::StringType>::Make(out_kernel);
    case arrow::Type::LARGE_STRING:
      return BinaryHashPartitionKernel<arrow::LargeStringType>::Make(out_kernel);
    case arrow::Type::BINARY:return BinaryHashPartitionKernel<arrow::BinaryType>::Make(out_kernel);
    case arrow::Type::LARGE_BINARY:
      return BinaryHashPartitionKernel<arrow::LargeBinaryType>::Make(out_kernel);
    case arrow::Type::FIXED_SIZE_BINARY:return FixedSizeBinaryHashPartitionKernel::Make(out_kernel);
    case arrow::Type::DATE32:return ModuloPartitionKernel<arrow::Date32Type>::Make(out_kernel);
    case arrow::Type::DATE64:return ModuloPartitionKernel<arrow::Date64Type>::Make(out_kernel);
    case arrow::Type::TIMESTAMP:return ModuloPartitionKernel<arrow::TimestampType>::Make(out_kernel);
    case arrow::Type::TIME32:return ModuloPartitionKernel<arrow::Time32Type>::Make(out_kernel);
    case arrow::Type::TIME64:return ModuloPartitionKernel<arrow::Time64Type>::Make(out_kernel);
    default:
      return {Code::NotImplemented,
              "Unsupported hash partition kernel data type" + data_type->ToString()};
  }
}

template<typename ARROW_T, typename = typename std::enable_if<
    arrow::is_number_type<ARROW_T>::value | arrow::is_boolean_type<ARROW_T>::value>::type>
class RangePartitionKernel : public PartitionKernel {
  using ArrayT = typename cylon::ArrowTypeTraits<ARROW_T>::ArrayT;
  using ScalarT = typename cylon::ArrowTypeTraits<ARROW_T>::ScalarT;
  using ValueT = typename cylon::ArrowTypeTraits<ARROW_T>::ValueT;

 public:
  RangePartitionKernel(const std::shared_ptr<CylonContext> &ctx,
                       bool ascending,
                       uint64_t num_samples,
                       uint32_t num_bins) : PartitionKernel(),
                                            ascending(ascending),
                                            num_bins(num_bins),
                                            num_samples(num_samples),
                                            ctx(ctx) {
  };

  static Status Make(const std::shared_ptr<CylonContext> &ctx, bool ascending, uint64_t num_samples,
                     uint32_t num_bins, std::unique_ptr<PartitionKernel> *out_kernel) {
    *out_kernel = std::make_unique<RangePartitionKernel<ARROW_T>>(ctx, ascending, num_samples,
                                                                  num_bins);
    return Status::OK();
  }

  Status Partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                   uint32_t num_partitions,
                   std::vector<uint32_t> &target_partitions,
                   std::vector<uint32_t> &partition_histogram) override {
    if (idx_col->null_count() > 0) {
      return {Code::Invalid, "Range partition kernel doesn't support null values"};
    }

    RETURN_CYLON_STATUS_IF_FAILED(build_bin_to_partition(idx_col, num_partitions));

    // resize vectors
    partition_histogram.resize(num_partitions, 0);
    target_partitions.resize(idx_col->length());

    return visit_chunked_array<ARROW_T>(
        idx_col,
        [&](uint64_t global_idx, ValueT val) {
          uint32_t p = ascending ?
                       bin_to_partition[get_bin_pos(val)] : num_partitions - 1
                           - bin_to_partition[get_bin_pos(val)];
          target_partitions[global_idx] = p;
          partition_histogram[p]++;
        },
        [&](uint64_t global_idx) {
          CYLON_UNUSED(global_idx);
        });
  }

 private:
  inline Status build_bin_to_partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                                       uint32_t num_partitions) {
    const std::shared_ptr<DataType> &data_type = tarrow::ToCylonType(idx_col->type());
    std::shared_ptr<arrow::ChunkedArray> sampled_array;

    // if len == num_samples, dont sample, just use idx col as it is!
    if ((uint64_t) idx_col->length() == num_samples) {
      sampled_array = std::make_shared<arrow::ChunkedArray>(idx_col->chunks());
    } else { // else, sample idx_col for num_samples
      std::shared_ptr<arrow::Array> samples;
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(util::SampleArray(idx_col,
                                                            num_samples,
                                                            samples,
                                                            ToArrowPool(ctx)));
      sampled_array = std::make_shared<arrow::ChunkedArray>(std::move(samples));
    }

    const auto &comm = ctx->GetCommunicator();

    // calculate minmax of the sample
    // todo: Allreduce MIN {min, -max} array as an optimization. Bool could be a problem.
    //   might need some template tricks

    std::shared_ptr<Scalar> min_scalar, max_scalar;
    if (sampled_array->length() == 0) {
      // if no samples present, set min and max to opposite ends so that Allreduce will determine the
      // correct value
      min_scalar = Scalar::Make(arrow::MakeScalar(std::numeric_limits<ValueT>::max()));
      max_scalar = Scalar::Make(arrow::MakeScalar(std::numeric_limits<ValueT>::min()));;
    } else {
      arrow::compute::ScalarAggregateOptions opts(true, 0);
      CYLON_ASSIGN_OR_RAISE(const auto &min_max, arrow::compute::MinMax(sampled_array, opts))
      const auto &min_max_v = min_max.scalar_as<arrow::StructScalar>().value;
      min_scalar = Scalar::Make(min_max_v[0]);
      max_scalar = Scalar::Make(min_max_v[1]);
    }

    std::shared_ptr<Scalar> global_min_scalar, global_max_scalar;
    RETURN_CYLON_STATUS_IF_FAILED(comm->AllReduce(min_scalar, net::MIN, &global_min_scalar));
    RETURN_CYLON_STATUS_IF_FAILED(comm->AllReduce(max_scalar, net::MAX, &global_max_scalar));

    min = std::static_pointer_cast<ScalarT>(global_min_scalar->data())->value;
    max = std::static_pointer_cast<ScalarT>(global_max_scalar->data())->value;
    range = max - min;

    // create sample histogram
    std::vector<uint64_t> local_counts(num_bins + 2, 0);
    for (const auto &arr: sampled_array->chunks()) {
      arrow::VisitArraySpanInline<ARROW_T>(*arr->data(),
                                           [&](ValueT val) {
                                             local_counts[get_bin_pos(val)]++;
                                           },
                                           []() {
                                             // idx_col is guaranteed to be non-null
                                           });
    }

    // all reduce local sample histograms
    const uint64_t *global_counts_ptr;
    std::shared_ptr<Column> global_counts;
    if (ctx->GetWorldSize() > 1) { // if distributed, all-reduce all local bin counts
      arrow::BufferVector buf{nullptr, arrow::Buffer::Wrap(local_counts)};
      const auto
          &array_data = arrow::ArrayData::Make(arrow::uint64(), num_bins + 2, std::move(buf), 0);
      auto local_counts_col = Column::Make(arrow::MakeArray(array_data));

      RETURN_CYLON_STATUS_IF_FAILED(comm->AllReduce(local_counts_col, net::SUM, &global_counts));
      global_counts_ptr =
          std::static_pointer_cast<arrow::UInt64Array>(global_counts->data())->raw_values();
    } else { // else, just use local bin counts
      global_counts_ptr = local_counts.data();
    }

    float_t quantile = float(1.0 / num_partitions), prefix_sum = 0;

//    LOG(INFO) << "len=" << idx_col->length() << " min=" << min << " max=" << max << " range=" <<
//              range << " num bins=" << num_bins << " quantile=" << quantile;

    // divide global histogram into quantiles
    const uint64_t total_samples = ctx->GetWorldSize() * num_samples;
    uint32_t curr_partition = 0;
    float_t target_quantile = quantile;
    for (uint32_t i = 0; i < num_bins + 2; i++) {
      bin_to_partition.push_back(curr_partition);
      float_t freq = (float_t) global_counts_ptr[i] / total_samples;
      prefix_sum += freq;
      if (prefix_sum > target_quantile) {
        curr_partition += (curr_partition
            < num_partitions - 1); // if curr_partition < numpartition: curr_partition++
        target_quantile += quantile;
      }
    }

    if (curr_partition != num_partitions - 1) {
      LOG(WARNING) << "sample was unable to capture distribution. try increasing num_samples or num_bins";
    }
    return Status::OK();
  }

  inline constexpr size_t get_bin_pos(ValueT val) {
    return (val >= min)
        + ((val >= min) & (val < max)) * ((val - min) * num_bins / range)
        + (val >= max) * num_bins;
  }

  const bool ascending;
  const uint32_t num_bins;
  const uint64_t num_samples;
  const std::shared_ptr<CylonContext> &ctx;
  std::vector<uint32_t> bin_to_partition;
  ValueT min, max, range;
};

Status CreateRangePartitionKernel(const std::shared_ptr<arrow::DataType> &data_type,
                                  const std::shared_ptr<CylonContext> &ctx,
                                  bool ascending,
                                  uint64_t num_samples,
                                  uint32_t num_bins,
                                  std::unique_ptr<PartitionKernel> *out_kernel) {
  switch (data_type->id()) {
    case arrow::Type::BOOL:
      return RangePartitionKernel<arrow::BooleanType>::Make(ctx, ascending, num_samples, num_bins,
                                                            out_kernel);
    case arrow::Type::UINT8:
      return RangePartitionKernel<arrow::UInt8Type>::Make(ctx, ascending, num_samples, num_bins,
                                                          out_kernel);
    case arrow::Type::INT8:
      return RangePartitionKernel<arrow::Int8Type>::Make(ctx, ascending, num_samples, num_bins,
                                                         out_kernel);
    case arrow::Type::UINT16:
      return RangePartitionKernel<arrow::UInt16Type>::Make(ctx, ascending, num_samples, num_bins,
                                                           out_kernel);
    case arrow::Type::INT16:
      return RangePartitionKernel<arrow::Int16Type>::Make(ctx, ascending, num_samples, num_bins,
                                                          out_kernel);
    case arrow::Type::UINT32:
      return RangePartitionKernel<arrow::UInt32Type>::Make(ctx, ascending, num_samples, num_bins,
                                                           out_kernel);
    case arrow::Type::INT32:
      return RangePartitionKernel<arrow::Int32Type>::Make(ctx, ascending, num_samples, num_bins,
                                                          out_kernel);
    case arrow::Type::UINT64:
      return RangePartitionKernel<arrow::UInt64Type>::Make(ctx, ascending, num_samples, num_bins,
                                                           out_kernel);
    case arrow::Type::INT64:
      return RangePartitionKernel<arrow::Int64Type>::Make(ctx, ascending, num_samples, num_bins,
                                                          out_kernel);
    case arrow::Type::FLOAT:
      return RangePartitionKernel<arrow::FloatType>::Make(ctx, ascending, num_samples, num_bins,
                                                          out_kernel);
    case arrow::Type::DOUBLE:
      return RangePartitionKernel<arrow::DoubleType>::Make(ctx, ascending, num_samples, num_bins,
                                                           out_kernel);
    default:
      return {Code::Invalid, "Range partition kernel does not support " + data_type->ToString()};
  }
}

RowHashingKernel::RowHashingKernel(const std::vector<std::shared_ptr<arrow::Field>> &fields) {
  for (auto const &field: fields) {
    std::unique_ptr<HashPartitionKernel> kernel;
    auto status = CreateHashPartitionKernel(field->type(), &kernel);
    if (!status.is_ok()) {
      throw std::runtime_error(status.get_msg());
    }
    this->hash_kernels.emplace_back(std::move(kernel));
  }
}

int32_t RowHashingKernel::Hash(const std::shared_ptr<arrow::Table> &table, int64_t row) const {
  if (table->num_rows() == 0) {
    return 0;
  }

  int32_t hash_code = 1;
  for (int c = 0; c < table->num_columns(); ++c) {
    hash_code = static_cast<int32_t>(31 * hash_code + this->hash_kernels[c]->ToHash(table->column(c)->chunk(0), row));
  }
  return hash_code;
}

}  // namespace cylon

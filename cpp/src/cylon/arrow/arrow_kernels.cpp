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
#include <type_traits>
#include <arrow/visit_data_inline.h>

#include <cylon/arrow/arrow_kernels.hpp>
#include <cylon/util/macros.hpp>
#include <cylon/util/sort.hpp>
#include <cylon/util/arrow_utils.hpp>
#include <cylon/arrow/arrow_comparator.hpp>
#include <utility>

namespace cylon {

inline Status create_output_arrays(uint32_t count, const std::shared_ptr<arrow::DataType> &type,
                                   std::vector<std::shared_ptr<arrow::Buffer>> buffers,
                                   int64_t null_count, int64_t offset,
                                   std::vector<std::shared_ptr<arrow::Array>> &output) {
  const auto &data = arrow::ArrayData::Make(type, count, std::move(buffers), null_count, offset);
  output.push_back(arrow::MakeArray(data));
  return Status::OK();
}

inline bool is_nulls_present(const std::shared_ptr<arrow::ChunkedArray> &values) {
  return values->null_count() > 0;
}

Status build_null_buffers_from_array(const std::shared_ptr<arrow::Array> &array,
                                     const std::vector<uint32_t> &target_partitions,
                                     const std::vector<uint32_t> &additional_counts,
                                     std::vector<arrow::TypedBufferBuilder<bool>> &null_bitmap_builders) {
  for (size_t i = 0; i < additional_counts.size(); i++) {
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(null_bitmap_builders[i].Reserve(additional_counts[i]));
  }

  for (int64_t i = 0; i < array->length(); i++) {
    const auto &target_buffer = target_partitions[i];
    null_bitmap_builders[target_buffer].UnsafeAppend(!array->IsNull(i));
  }
  return Status::OK();
}

Status build_null_buffers(const std::shared_ptr<arrow::ChunkedArray> &values,
                          std::vector<arrow::TypedBufferBuilder<bool>> &null_bitmap_builders,
                          const std::vector<uint32_t> &target_partitions) {
  int64_t offset = 0;
  for (const auto &array: values->chunks()) {
    const int64_t arr_len = array->length();
    for (int64_t i = 0; i < arr_len; i++) {
      const auto &target_buffer = target_partitions[offset + i];
      null_bitmap_builders[target_buffer].UnsafeAppend(!array->IsNull(i));
    }
    offset += arr_len;
  }
  assert(offset == values->length());
  return Status::OK();
}
// SPLITTING -----------------------------------------------------------------------------

template<typename TYPE,
    typename = typename std::enable_if<arrow::is_number_type<TYPE>::value | arrow::is_boolean_type<TYPE>::value
                                           | arrow::is_temporal_type<TYPE>::value>::type>
class ArrowArrayNumericSplitKernel : public ArrowArraySplitKernel {
 public:
  explicit ArrowArrayNumericSplitKernel(arrow::MemoryPool *pool) : ArrowArraySplitKernel(pool) {}
  using T = typename TYPE::c_type;

  Status Split(const std::shared_ptr<arrow::ChunkedArray> &values, uint32_t num_partitions,
               const std::vector<uint32_t> &target_partitions, const std::vector<uint32_t> &counts,
               std::vector<std::shared_ptr<arrow::Array>> &output) override {
    if ((size_t) values->length() != target_partitions.size()) {
      return {Code::ExecutionError, "values rows != target_partitions length"};
    }

    std::vector<std::shared_ptr<arrow::Buffer>> build_buffers;
    std::vector<arrow::TypedBufferBuilder<bool>> null_bitmap_builders;

    std::vector<T *> data_buffers;
    data_buffers.reserve(num_partitions);
    std::vector<int> buffer_indexes;
    buffer_indexes.reserve(num_partitions);

    bool nulls_present = is_nulls_present(values);
    for (uint32_t i = 0; i < num_partitions; i++) {
      int64_t buf_size = counts[i] * sizeof(T);
      arrow::Result<std::unique_ptr<arrow::Buffer>> result = arrow::AllocateBuffer(buf_size, pool_);
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
      build_buffers.push_back(std::move(result.ValueOrDie()));
      auto *indices_begin = reinterpret_cast<T *>(build_buffers.back()->mutable_data());
      data_buffers.push_back(indices_begin);
      buffer_indexes.push_back(0);
      if (nulls_present) {
        null_bitmap_builders.emplace_back(pool_);
        RETURN_CYLON_STATUS_IF_ARROW_FAILED(null_bitmap_builders.back().Reserve(counts[i]));
      }
    }

    size_t offset = 0;
    for (const auto &array: values->chunks()) {
      const std::shared_ptr<arrow::ArrayData> &data = array->data();
      const T *value_buffer = data->template GetValues<T>(1);

      const int64_t arr_len = array->length();
      for (int64_t i = 0; i < arr_len; i++, offset++) {
        uint32_t target_buffer = target_partitions[offset];
        int idx = buffer_indexes[target_buffer];
        data_buffers[target_buffer][idx] = value_buffer[i];
        buffer_indexes[target_buffer] = idx + 1;
      }
    }

    if (nulls_present) {
      RETURN_CYLON_STATUS_IF_FAILED(build_null_buffers(values, null_bitmap_builders, target_partitions));
    }

    output.reserve(num_partitions);
    if (!nulls_present) {
      for (uint32_t i = 0; i < num_partitions; i++) {
        RETURN_CYLON_STATUS_IF_FAILED(create_output_arrays(counts[i],
                                                           values->type(),
                                                           {nullptr, build_buffers[i]},
                                                           0,
                                                           0,
                                                           output));
      }
    } else {
      for (uint32_t i = 0; i < num_partitions; i++) {
        std::shared_ptr<arrow::Buffer> null_buffer;
        int64_t null_count = null_bitmap_builders[i].false_count();
        RETURN_CYLON_STATUS_IF_ARROW_FAILED(null_bitmap_builders[i].Finish(&null_buffer));
        RETURN_CYLON_STATUS_IF_FAILED(create_output_arrays(counts[i],
                                                           values->type(),
                                                           {std::move(null_buffer), build_buffers[i]},
                                                           null_count,
                                                           0,
                                                           output));
      }
    }
    return Status::OK();
  }
};

class FixedBinaryArraySplitKernel : public ArrowArraySplitKernel {
 public:
  using ARROW_ARRAY_T = arrow::FixedSizeBinaryArray;

  explicit FixedBinaryArraySplitKernel(arrow::MemoryPool *pool) : ArrowArraySplitKernel(pool) {}

  Status Split(const std::shared_ptr<arrow::ChunkedArray> &values, uint32_t num_partitions,
               const std::vector<uint32_t> &target_partitions, const std::vector<uint32_t> &counts,
               std::vector<std::shared_ptr<arrow::Array>> &output) override {
    if ((size_t) values->length() != target_partitions.size()) {
      return {Code::ExecutionError, "values rows != target_partitions length"};
    }

    // if chunks 0 we can return immediately
    if (values->chunks().empty()) {
      return Status::OK();
    }

    std::shared_ptr<ARROW_ARRAY_T> first_array = std::static_pointer_cast<ARROW_ARRAY_T>(values->chunk(0));
    int32_t width = first_array->byte_width();
    std::vector<arrow::TypedBufferBuilder<bool>> null_bitmap_builders;
    std::vector<std::shared_ptr<arrow::Buffer>> build_buffers;
    std::vector<uint8_t *> data_buffers;
    std::vector<int> buffer_indexes;

    bool nulls_present = is_nulls_present(values);
    for (uint32_t i = 0; i < num_partitions; i++) {
      int64_t buf_size = counts[i] * width;
      arrow::Result<std::unique_ptr<arrow::Buffer>> result = arrow::AllocateBuffer(buf_size, pool_);
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
      build_buffers.push_back(std::move(result.ValueOrDie()));
      uint8_t *indices_begin = build_buffers.back()->mutable_data();
      data_buffers.push_back(indices_begin);
      buffer_indexes.push_back(0);
      if (nulls_present) {
        null_bitmap_builders.emplace_back(pool_);
        RETURN_CYLON_STATUS_IF_ARROW_FAILED(null_bitmap_builders.back().Reserve(counts[i]));
      }
    }

    size_t offset = 0;
    for (const auto &array: values->chunks()) {
      const std::shared_ptr<arrow::ArrayData> &data = array->data();
      const auto *value_buffer = data->template GetValues<uint8_t>(1);
      const int64_t arr_len = array->length();
      for (int64_t i = 0; i < arr_len; i++, offset++) {
        unsigned int target_buffer = target_partitions[offset];
        int idx = buffer_indexes[target_buffer];
        memcpy(data_buffers[target_buffer] + idx * width, value_buffer + i * width, width);
        buffer_indexes[target_buffer] = idx + 1;
      }
    }

    if (nulls_present) {
      RETURN_CYLON_STATUS_IF_FAILED(build_null_buffers(values, null_bitmap_builders, target_partitions));
    }

    output.reserve(num_partitions);
    if (!nulls_present) {
      for (uint32_t i = 0; i < num_partitions; i++) {
        create_output_arrays(counts[i], values->type(), {nullptr, build_buffers[i]}, 0, 0, output);
      }
    } else {
      for (uint32_t i = 0; i < num_partitions; i++) {
        std::shared_ptr<arrow::Buffer> null_buffer;
        int64_t null_count = null_bitmap_builders[i].false_count();
        RETURN_CYLON_STATUS_IF_ARROW_FAILED(null_bitmap_builders[i].Finish(&null_buffer));
        create_output_arrays(counts[i], values->type(), {null_buffer, build_buffers[i]}, null_count, 0, output);
      }
    }
    return Status::OK();
  }
};

template<typename TYPE>
class BinaryArraySplitKernel : public ArrowArraySplitKernel {
 public:
  using ARROW_ARRAY_T = typename arrow::TypeTraits<TYPE>::ArrayType;
  using ARROW_BUILDER_T = typename arrow::TypeTraits<TYPE>::BuilderType;
  using ARROW_OFFSET_T = typename TYPE::offset_type;

  explicit BinaryArraySplitKernel(arrow::MemoryPool *pool) : ArrowArraySplitKernel(pool) {}

  Status Split(const std::shared_ptr<arrow::ChunkedArray> &values, uint32_t num_partitions,
               const std::vector<uint32_t> &target_partitions, const std::vector<uint32_t> &counts,
               std::vector<std::shared_ptr<arrow::Array>> &output) override {
    CYLON_UNUSED(counts);
    if ((size_t) values->length() != target_partitions.size()) {
      return {Code::ExecutionError, "values rows != target_partitions length"};
    }
    std::vector<std::shared_ptr<arrow::TypedBufferBuilder<bool>>> null_bitmap_builders;
    bool nulls_present = is_nulls_present(values);

    std::vector<std::unique_ptr<ARROW_BUILDER_T>> builders;
    builders.reserve(num_partitions);
    for (uint32_t i = 0; i < num_partitions; i++) {
      builders.emplace_back(std::make_unique<ARROW_BUILDER_T>(pool_));
    }

    size_t offset = 0;
    for (const auto &array: values->chunks()) {
      std::shared_ptr<ARROW_ARRAY_T> casted_array = std::static_pointer_cast<ARROW_ARRAY_T>(array);
      const int64_t arr_len = array->length();
      for (int64_t i = 0; i < arr_len; i++, offset++) {
        ARROW_OFFSET_T length = 0;
        if (nulls_present && casted_array->IsNull(i)) {
          RETURN_CYLON_STATUS_IF_ARROW_FAILED(builders[target_partitions[offset]]->AppendNull());
        } else {
          const uint8_t *value = casted_array->GetValue(i, &length);
          RETURN_CYLON_STATUS_IF_ARROW_FAILED(builders[target_partitions[offset]]->Append(value, length));
        }
      }
    }

    output.reserve(num_partitions);
    for (uint32_t i = 0; i < num_partitions; i++) {
      std::shared_ptr<arrow::Array> array;
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(builders[i]->Finish(&array));
      output.push_back(array);
    }

    return Status::OK();
  }
};

using UInt8ArraySplitter = ArrowArrayNumericSplitKernel<arrow::UInt8Type>;
using UInt16ArraySplitter = ArrowArrayNumericSplitKernel<arrow::UInt16Type>;
using UInt32ArraySplitter = ArrowArrayNumericSplitKernel<arrow::UInt32Type>;
using UInt64ArraySplitter = ArrowArrayNumericSplitKernel<arrow::UInt64Type>;
using Int8ArraySplitter = ArrowArrayNumericSplitKernel<arrow::Int8Type>;
using Int16ArraySplitter = ArrowArrayNumericSplitKernel<arrow::Int16Type>;
using Int32ArraySplitter = ArrowArrayNumericSplitKernel<arrow::Int32Type>;
using Int64ArraySplitter = ArrowArrayNumericSplitKernel<arrow::Int64Type>;

using HalfFloatArraySplitter = ArrowArrayNumericSplitKernel<arrow::HalfFloatType>;
using FloatArraySplitter = ArrowArrayNumericSplitKernel<arrow::FloatType>;
using DoubleArraySplitter = ArrowArrayNumericSplitKernel<arrow::DoubleType>;

std::unique_ptr<ArrowArraySplitKernel> CreateSplitter(const std::shared_ptr<arrow::DataType> &type,
                                                      arrow::MemoryPool *pool) {
  switch (type->id()) {
    case arrow::Type::UINT8: return std::make_unique<UInt8ArraySplitter>(pool);
    case arrow::Type::INT8: return std::make_unique<Int8ArraySplitter>(pool);
    case arrow::Type::UINT16: return std::make_unique<UInt16ArraySplitter>(pool);
    case arrow::Type::INT16: return std::make_unique<Int16ArraySplitter>(pool);
    case arrow::Type::UINT32:return std::make_unique<UInt32ArraySplitter>(pool);
    case arrow::Type::INT32:return std::make_unique<Int32ArraySplitter>(pool);
    case arrow::Type::UINT64:return std::make_unique<UInt64ArraySplitter>(pool);
    case arrow::Type::INT64:return std::make_unique<Int64ArraySplitter>(pool);
    case arrow::Type::FLOAT:return std::make_unique<FloatArraySplitter>(pool);
    case arrow::Type::DOUBLE:return std::make_unique<DoubleArraySplitter>(pool);
    case arrow::Type::FIXED_SIZE_BINARY:return std::make_unique<FixedBinaryArraySplitKernel>(pool);
    case arrow::Type::STRING:return std::make_unique<BinaryArraySplitKernel<arrow::StringType>>(pool);
    case arrow::Type::BINARY:return std::make_unique<BinaryArraySplitKernel<arrow::BinaryType>>(pool);
    case arrow::Type::DATE32:return std::make_unique<ArrowArrayNumericSplitKernel<arrow::Date32Type>>(pool);
    case arrow::Type::DATE64:return std::make_unique<ArrowArrayNumericSplitKernel<arrow::Date64Type>>(pool);
    case arrow::Type::TIMESTAMP:return std::make_unique<ArrowArrayNumericSplitKernel<arrow::TimestampType>>(pool);
    case arrow::Type::TIME32:return std::make_unique<ArrowArrayNumericSplitKernel<arrow::Time32Type>>(pool);
    case arrow::Type::TIME64:return std::make_unique<ArrowArrayNumericSplitKernel<arrow::Time64Type>>(pool);
    default:return nullptr;
  }
}

// SORTING -----------------------------------------------------------------------------

template<typename ARROW_T>
class ArrowBinarySortKernel : public IndexSortKernel {
  using ARRAY_T = typename arrow::TypeTraits<ARROW_T>::ArrayType;

 public:
  ArrowBinarySortKernel(arrow::MemoryPool *pool, bool ascending)
      : IndexSortKernel(pool, ascending) {}

  arrow::Status Sort(const std::shared_ptr<arrow::Array> &values,
                     std::shared_ptr<arrow::UInt64Array> &offsets) const override {
    auto array = std::static_pointer_cast<ARRAY_T>(values);

    if (ascending) {
      return do_sort(
          [&array](uint64_t left, uint64_t right) {
            return array->GetView(left).compare(array->GetView(right)) < 0;
          },
          values->length(), pool_, offsets);
    } else {
      return do_sort(
          [&array](uint64_t left, uint64_t right) {
            return array->GetView(left).compare(array->GetView(right)) > 0;
          },
          values->length(), pool_, offsets);
    }
  }
};

template<typename TYPE,
    typename = typename std::enable_if<arrow::is_number_type<TYPE>::value | arrow::is_boolean_type<TYPE>::value
                                           | arrow::is_temporal_type<TYPE>::value>::type>
class NumericIndexSortKernel : public IndexSortKernel {
 public:
  using T = typename TYPE::c_type;

  NumericIndexSortKernel(arrow::MemoryPool *pool, bool ascending)
      : IndexSortKernel(pool, ascending) {}

  arrow::Status Sort(const std::shared_ptr<arrow::Array> &values,
                     std::shared_ptr<arrow::UInt64Array> &offsets) const override {
    auto array = std::static_pointer_cast<arrow::NumericArray<TYPE>>(values);

    const T *left_data = array->raw_values();

    if (ascending) {
      return do_sort([&left_data](uint64_t left,
                                  uint64_t right) { return left_data[left] < left_data[right]; },
                     values->length(), pool_, offsets);
    } else {
      return do_sort([&left_data](uint64_t left,
                                  uint64_t right) { return left_data[left] > left_data[right]; },
                     values->length(), pool_, offsets);
    }
  }
};

using UInt8ArraySorter = NumericIndexSortKernel<arrow::UInt8Type>;
using UInt16ArraySorter = NumericIndexSortKernel<arrow::UInt16Type>;
using UInt32ArraySorter = NumericIndexSortKernel<arrow::UInt32Type>;
using UInt64ArraySorter = NumericIndexSortKernel<arrow::UInt64Type>;
using Int8ArraySorter = NumericIndexSortKernel<arrow::Int8Type>;
using Int16ArraySorter = NumericIndexSortKernel<arrow::Int16Type>;
using Int32ArraySorter = NumericIndexSortKernel<arrow::Int32Type>;
using Int64ArraySorter = NumericIndexSortKernel<arrow::Int64Type>;
using HalfFloatArraySorter = NumericIndexSortKernel<arrow::HalfFloatType>;
using FloatArraySorter = NumericIndexSortKernel<arrow::FloatType>;
using DoubleArraySorter = NumericIndexSortKernel<arrow::DoubleType>;

std::unique_ptr<IndexSortKernel> CreateSorter(const std::shared_ptr<arrow::DataType> &type,
                                              arrow::MemoryPool *pool, bool ascending) {
  switch (type->id()) {
    case arrow::Type::UINT8:return std::make_unique<UInt8ArraySorter>(pool, ascending);
    case arrow::Type::INT8:return std::make_unique<Int8ArraySorter>(pool, ascending);
    case arrow::Type::UINT16:return std::make_unique<UInt16ArraySorter>(pool, ascending);
    case arrow::Type::INT16:return std::make_unique<Int16ArraySorter>(pool, ascending);
    case arrow::Type::UINT32:return std::make_unique<UInt32ArraySorter>(pool, ascending);
    case arrow::Type::INT32:return std::make_unique<Int32ArraySorter>(pool, ascending);
    case arrow::Type::UINT64:return std::make_unique<UInt64ArraySorter>(pool, ascending);
    case arrow::Type::INT64:return std::make_unique<Int64ArraySorter>(pool, ascending);
    case arrow::Type::FLOAT:return std::make_unique<FloatArraySorter>(pool, ascending);
    case arrow::Type::DOUBLE:return std::make_unique<DoubleArraySorter>(pool, ascending);
    case arrow::Type::STRING:return std::make_unique<ArrowBinarySortKernel<arrow::StringType>>(pool, ascending);
    case arrow::Type::BINARY:return std::make_unique<ArrowBinarySortKernel<arrow::BinaryType>>(pool, ascending);
    case arrow::Type::FIXED_SIZE_BINARY:
      return std::make_unique<ArrowBinarySortKernel<arrow::FixedSizeBinaryType>>(pool,
                                                                                 ascending);
    case arrow::Type::DATE32:return std::make_unique<NumericIndexSortKernel<arrow::Date32Type>>(pool, ascending);
    case arrow::Type::DATE64:return std::make_unique<NumericIndexSortKernel<arrow::Date64Type>>(pool, ascending);
    case arrow::Type::TIMESTAMP:return std::make_unique<NumericIndexSortKernel<arrow::TimestampType>>(pool, ascending);
    case arrow::Type::TIME32:return std::make_unique<NumericIndexSortKernel<arrow::Time32Type>>(pool, ascending);
    case arrow::Type::TIME64:return std::make_unique<NumericIndexSortKernel<arrow::Time64Type>>(pool, ascending);
    default:return nullptr;
  }
}

arrow::Status SortIndices(arrow::MemoryPool *memory_pool,
                          const std::shared_ptr<arrow::Array> &values,
                          std::shared_ptr<arrow::UInt64Array> &offsets, bool ascending) {
  std::unique_ptr<IndexSortKernel> out = CreateSorter(values->type(), memory_pool, ascending);
  if (out == nullptr) {
    return arrow::Status::NotImplemented("unknown type " + values->type()->ToString());
  }
  return out->Sort(values, offsets);
}

template<typename TYPE,
    typename = typename std::enable_if<arrow::is_number_type<TYPE>::value | arrow::is_boolean_type<TYPE>::value
                                           | arrow::is_temporal_type<TYPE>::value>::type>
class NumericInplaceIndexSortKernel : public InplaceIndexSortKernel {
 public:
  using T = typename TYPE::c_type;

  explicit NumericInplaceIndexSortKernel(arrow::MemoryPool *pool) : InplaceIndexSortKernel(pool) {}

  arrow::Status Sort(std::shared_ptr<arrow::Array> &values,
                     std::shared_ptr<arrow::UInt64Array> &offsets) override {
    auto array = std::static_pointer_cast<arrow::NumericArray<TYPE>>(values);
    const std::shared_ptr<arrow::ArrayData> &data = array->data();
    // get the first buffer as a mutable buffer
    if (!util::IsMutable(values)) {
      return arrow::Status::ExecutionError("inplace sort called on an array with immutable buffers");
    }

    T *left_data = data->template GetMutableValues<T>(1);
    int64_t length = values->length();
    int64_t buf_size = length * (int64_t) (sizeof(uint64_t));

    arrow::Result<std::unique_ptr<arrow::Buffer>> result = AllocateBuffer(buf_size, pool_);
    RETURN_ARROW_STATUS_IF_FAILED(result.status());
    std::shared_ptr<arrow::Buffer> indices_buf(std::move(result.ValueOrDie()));

    auto *indices_begin = reinterpret_cast<int64_t *>(indices_buf->mutable_data());
    for (int64_t i = 0; i < length; i++) {
      indices_begin[i] = i;
    }
    cylon::util::introsort(left_data, indices_begin, length);
    offsets = std::make_shared<arrow::UInt64Array>(length, indices_buf);
    return arrow::Status::OK();
  }
};

using UInt8ArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::UInt8Type>;
using UInt16ArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::UInt16Type>;
using UInt32ArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::UInt32Type>;
using UInt64ArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::UInt64Type>;
using Int8ArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::Int8Type>;
using Int16ArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::Int16Type>;
using Int32ArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::Int32Type>;
using Int64ArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::Int64Type>;
using HalfFloatArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::HalfFloatType>;
using FloatArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::FloatType>;
using DoubleArrayInplaceSorter = NumericInplaceIndexSortKernel<arrow::DoubleType>;

std::unique_ptr<InplaceIndexSortKernel> CreateInplaceSorter(
    const std::shared_ptr<arrow::DataType> &type, arrow::MemoryPool *pool) {
  switch (type->id()) {
    case arrow::Type::UINT8: return std::make_unique<UInt8ArrayInplaceSorter>(pool);
    case arrow::Type::INT8: return std::make_unique<Int8ArrayInplaceSorter>(pool);
    case arrow::Type::UINT16: return std::make_unique<UInt16ArrayInplaceSorter>(pool);
    case arrow::Type::INT16: return std::make_unique<Int16ArrayInplaceSorter>(pool);
    case arrow::Type::UINT32: return std::make_unique<UInt32ArrayInplaceSorter>(pool);
    case arrow::Type::INT32: return std::make_unique<Int32ArrayInplaceSorter>(pool);
    case arrow::Type::UINT64:return std::make_unique<UInt64ArrayInplaceSorter>(pool);
    case arrow::Type::INT64:return std::make_unique<Int64ArrayInplaceSorter>(pool);
    case arrow::Type::FLOAT:return std::make_unique<FloatArrayInplaceSorter>(pool);
    case arrow::Type::DOUBLE:return std::make_unique<DoubleArrayInplaceSorter>(pool);
    case arrow::Type::DATE32:return std::make_unique<NumericInplaceIndexSortKernel<arrow::Date32Type>>(pool);
    case arrow::Type::DATE64:return std::make_unique<NumericInplaceIndexSortKernel<arrow::Date64Type>>(pool);
    case arrow::Type::TIMESTAMP:return std::make_unique<NumericInplaceIndexSortKernel<arrow::TimestampType>>(pool);
    case arrow::Type::TIME32:return std::make_unique<NumericInplaceIndexSortKernel<arrow::Time32Type>>(pool);
    case arrow::Type::TIME64:return std::make_unique<NumericInplaceIndexSortKernel<arrow::Time64Type>>(pool);
    default:return nullptr;
  }
}

arrow::Status SortIndicesInPlace(arrow::MemoryPool *memory_pool,
                                 std::shared_ptr<arrow::Array> &values,
                                 std::shared_ptr<arrow::UInt64Array> &offsets) {
  std::unique_ptr<InplaceIndexSortKernel> out = CreateInplaceSorter(values->type(), memory_pool);
  if (out == nullptr) {
    return arrow::Status::NotImplemented("unknown type " + values->type()->ToString());
  }
  return out->Sort(values, offsets);
}

template<typename Comparator>
arrow::Status do_sort(Comparator &&comp, int64_t len, arrow::MemoryPool *pool,
                      std::shared_ptr<arrow::UInt64Array> &offsets) {
  auto buf_size = static_cast<int64_t>(len * sizeof(int64_t));

  arrow::Result<std::unique_ptr<arrow::Buffer>> result = arrow::AllocateBuffer(buf_size, pool);
  RETURN_ARROW_STATUS_IF_FAILED(result.status());
  std::shared_ptr<arrow::Buffer> indices_buf(std::move(result).ValueOrDie());

  auto *indices_begin = reinterpret_cast<int64_t *>(indices_buf->mutable_data());
  for (int64_t i = 0; i < len; i++) {
    indices_begin[i] = i;
  }

  int64_t *indices_end = indices_begin + len;
  std::sort(indices_begin, indices_end, comp);
  offsets = std::make_shared<arrow::UInt64Array>(len, indices_buf);
  return arrow::Status::OK();
}

// MULTI COLUMN SORTING -----------------------------------------------------------------------

arrow::Status SortIndicesMultiColumns(arrow::MemoryPool *memory_pool,
                                      const std::shared_ptr<arrow::Table> &table,
                                      const std::vector<int32_t> &columns,
                                      std::shared_ptr<arrow::UInt64Array> &offsets,
                                      const std::vector<bool> &ascending) {
  if (columns.size() != ascending.size()) {
    return arrow::Status::Invalid("No of sort columns and no of sort direction indicators mismatch");
  }

  if (util::CheckArrowTableContainsChunks(table, columns)){
    return arrow::Status::Invalid("SortIndicesMultiColumns can not handle chunked columns");
  }

  std::vector<std::shared_ptr<ArrayIndexComparator>> comparators;
  comparators.reserve(columns.size());
  for (size_t i = 0; i < columns.size(); i++) {
    std::unique_ptr<ArrayIndexComparator> comp;
    auto status =
        CreateArrayIndexComparator(cylon::util::GetChunkOrEmptyArray(table->column(columns[i]), 0),
                                   &comp, ascending[i], /*null_order=*/true);
    if (!status.is_ok()) {
      return arrow::Status::Invalid(status.get_msg());
    }
    comparators.emplace_back(std::move(comp));
  }

  auto buf_size = (int64_t) (table->num_rows() * sizeof(int64_t));

  arrow::Result<std::unique_ptr<arrow::Buffer>> result = arrow::AllocateBuffer(buf_size, memory_pool);
  const arrow::Status &status = result.status();
  if (!status.ok()) {
    LOG(FATAL) << "Failed to allocate sort indices - " << status.message();
    return status;
  }
  std::shared_ptr<arrow::Buffer> indices_buf(std::move(result.ValueOrDie()));

  auto *indices_begin = reinterpret_cast<int64_t *>(indices_buf->mutable_data());
  for (int64_t i = 0; i < table->num_rows(); i++) {
    indices_begin[i] = i;
  }

  int64_t *indices_end = indices_begin + table->num_rows();
  std::sort(indices_begin, indices_end, [&comparators](int64_t idx1, int64_t idx2) {
    for (auto const &comp: comparators) {
      auto res = comp->compare(idx1, idx2);
      if (res != 0) {
        return res < 0;
      }
    }
    return false; // if this point is reached, that means every comparison has returned 0! so, equal. i.e. NOT less
  });

  offsets = std::make_shared<arrow::UInt64Array>(table->num_rows(), indices_buf);
  return arrow::Status::OK();
}

arrow::Status SortIndicesMultiColumns(arrow::MemoryPool *memory_pool,
                                      const std::shared_ptr<arrow::Table> &table,
                                      const std::vector<int32_t> &columns,
                                      std::shared_ptr<arrow::UInt64Array> &offsets) {
  return SortIndicesMultiColumns(memory_pool, table, columns, offsets,
                                 std::vector<bool>(columns.size(), true));
}

// STREAMING SPLIT-----------------------------------------------------------------------------

template<typename TYPE>
class NumericStreamingSplitKernel : public StreamingSplitKernel {
  using T = typename TYPE::c_type;

 public:
  NumericStreamingSplitKernel(int32_t num_targets, arrow::MemoryPool *pool)
      : StreamingSplitKernel(), num_partitions(num_targets), pool_(pool) {
    build_buffers.reserve(num_targets);
    counts.reserve(num_targets);
    buffer_indexes.reserve(num_targets);
    for (int i = 0; i < num_targets; i++) {
      int64_t buf_size = 1024 * 1024 * sizeof(T);
      arrow::Result<std::unique_ptr<arrow::ResizableBuffer>> result = arrow::AllocateResizableBuffer(buf_size, pool_);
      build_buffers.push_back(std::move(result.ValueOrDie()));
      null_bitmap_builders.push_back(arrow::TypedBufferBuilder<bool>(pool_));
      counts.push_back(0);
      buffer_indexes.push_back(0);
    }
  }

  Status Split(const std::shared_ptr<arrow::Array> &values, const std::vector<uint32_t> &partitions,
               const std::vector<uint32_t> &cnts) override {
    type = values->type();
    // reserve additional space in the builders
    data_buffers.clear();
    for (uint32_t i = 0; i < num_partitions; i++) {
      counts[i] += cnts[i];
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(build_buffers[i]->template TypedResize<T>(counts[i], false));
      auto *indices_begin = reinterpret_cast<T *>(build_buffers[i]->mutable_data());
      data_buffers.push_back(indices_begin);
    }

    const std::shared_ptr<arrow::ArrayData> &data = values->data();
    const T *value_buffer = data->template GetValues<T>(1);
    const int64_t arr_len = values->length();
    for (int64_t i = 0; i < arr_len; i++) {
      unsigned int i1 = partitions[i];
      auto *indices_begin = data_buffers[i1];
      int idx = buffer_indexes[i1];
      indices_begin[idx] = value_buffer[i];
      buffer_indexes[i1] = buffer_indexes[i1] + 1;
    }

    if (values->null_count() > 0) {
      RETURN_CYLON_STATUS_IF_FAILED(build_null_buffers_from_array(values, partitions, cnts, null_bitmap_builders));
      nulls_present = true;
    }
    return Status::OK();
  }

  Status Finish(std::vector<std::shared_ptr<arrow::Array>> &output) override {
    output.reserve(num_partitions);
    if (!nulls_present) {
      for (uint32_t i = 0; i < num_partitions; i++) {
        RETURN_CYLON_STATUS_IF_FAILED(create_output_arrays(counts[i], type,
                                                           {nullptr, build_buffers[i]},
                                                           0, 0, output));
      }
    } else {
      for (uint32_t i = 0; i < num_partitions; i++) {
        std::shared_ptr<arrow::Buffer> null_buffer;
        int64_t null_count = null_bitmap_builders[i].false_count();
        RETURN_CYLON_STATUS_IF_ARROW_FAILED(null_bitmap_builders[i].Finish(&null_buffer));
        RETURN_CYLON_STATUS_IF_FAILED(create_output_arrays(counts[i], type,
                                                           {std::move(null_buffer), build_buffers[i]},
                                                           null_count, 0, output));
      }
    }
    return Status::OK();
  }

 private:
  uint32_t num_partitions;
  std::shared_ptr<arrow::DataType> type;
  std::vector<int> counts;
  std::vector<std::shared_ptr<arrow::ResizableBuffer>> build_buffers;
  std::vector<T *> data_buffers;
  std::vector<int> buffer_indexes;
  arrow::MemoryPool *pool_;
  std::vector<arrow::TypedBufferBuilder<bool>> null_bitmap_builders;
  bool nulls_present = false;
};

class FixedBinaryStreamingSplitKernel : public StreamingSplitKernel {
 public:
  FixedBinaryStreamingSplitKernel(const std::shared_ptr<arrow::DataType> &type_,
                                  int32_t num_targets, arrow::MemoryPool *pool)
      : StreamingSplitKernel(), builders_({}) {
    builders_.reserve(num_targets);
    for (int i = 0; i < num_targets; i++) {
      builders_.emplace_back(std::make_shared<arrow::FixedSizeBinaryBuilder>(type_, pool));
    }
  }

  Status Split(const std::shared_ptr<arrow::Array> &values, const std::vector<uint32_t> &partitions,
               const std::vector<uint32_t> &cnts) override {
    const auto &reader = std::static_pointer_cast<arrow::FixedSizeBinaryArray>(values);

    for (size_t i = 0; i < builders_.size(); i++) {
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(builders_[i]->Reserve(cnts[i]));
    }
    for (int64_t i = 0; i < values->length(); i++) {
      if (!reader->IsNull(i)) {
        const uint8_t *value = reader->Value(i);
        builders_[partitions[i]]->UnsafeAppend(value);
      } else {
        builders_[partitions[i]]->UnsafeAppendNull();
      }
    }
    return Status::OK();
  }

  Status Finish(std::vector<std::shared_ptr<arrow::Array>> &out) override {
    out.reserve(builders_.size());
    for (auto &&builder: builders_) {
      std::shared_ptr<arrow::Array> array;
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder->Finish(&array));
      out.emplace_back(std::move(array));
    }
    return Status::OK();
  }

 private:
  std::vector<std::shared_ptr<arrow::FixedSizeBinaryBuilder>> builders_;
};

template<typename TYPE>
class BinaryStreamingSplitKernel : public StreamingSplitKernel {
  using BUILDER_T = typename arrow::TypeTraits<TYPE>::BuilderType;
  using ARROW_OFFSET_T = typename TYPE::offset_type;

 public:
  BinaryStreamingSplitKernel(int32_t &targets, arrow::MemoryPool *pool)
      : StreamingSplitKernel(), builders_({}) {
    builders_.reserve(targets);
    for (int i = 0; i < targets; i++) {
      builders_.emplace_back(std::make_shared<BUILDER_T>(pool));
    }
  }

  Status Split(const std::shared_ptr<arrow::Array> &array, const std::vector<uint32_t> &partitions,
               const std::vector<uint32_t> &cnts) override {
    CYLON_UNUSED(cnts);
    auto reader = std::static_pointer_cast<arrow::BinaryArray>(array);
    const int64_t arr_len = array->length();
    for (int64_t i = 0; i < arr_len; i++) {
      if (!reader->IsNull(i)) {
        ARROW_OFFSET_T length = 0;
        const uint8_t *value = reader->GetValue(i, &length);
        RETURN_CYLON_STATUS_IF_ARROW_FAILED(builders_[partitions[i]]->Append(value, length));
      } else {
        RETURN_CYLON_STATUS_IF_ARROW_FAILED(builders_[partitions[i]]->AppendNull());
      }
    }
    return Status::OK();
  }

  Status Finish(std::vector<std::shared_ptr<arrow::Array>> &out) override {
    out.reserve(builders_.size());
    for (size_t i = 0; i < builders_.size(); i++) {
      std::shared_ptr<arrow::Array> array;
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(builders_[i]->Finish(&array));
      out.emplace_back(std::move(array));
    }
    return Status::OK();
  }

 private:
  std::vector<std::shared_ptr<BUILDER_T>> builders_;
};

using UInt8ArrayStreamingSplitter = NumericStreamingSplitKernel<arrow::UInt8Type>;
using UInt16ArrayStreamingSplitter = NumericStreamingSplitKernel<arrow::UInt16Type>;
using UInt32ArrayStreamingSplitter = NumericStreamingSplitKernel<arrow::UInt32Type>;
using UInt64ArrayStreamingSplitter = NumericStreamingSplitKernel<arrow::UInt64Type>;

using Int8ArrayStreamingSplitter = NumericStreamingSplitKernel<arrow::Int8Type>;
using Int16ArrayStreamingSplitter = NumericStreamingSplitKernel<arrow::Int16Type>;
using Int32ArrayStreamingSplitter = NumericStreamingSplitKernel<arrow::Int32Type>;
using Int64ArrayStreamingSplitter = NumericStreamingSplitKernel<arrow::Int64Type>;

using HalfFloatArrayStreamingSplitter = NumericStreamingSplitKernel<arrow::HalfFloatType>;
using FloatArrayStreamingSplitter = NumericStreamingSplitKernel<arrow::FloatType>;
using DoubleArrayStreamingSplitter = NumericStreamingSplitKernel<arrow::DoubleType>;

std::unique_ptr<StreamingSplitKernel> CreateStreamingSplitter(
    const std::shared_ptr<arrow::DataType> &type, int32_t targets, arrow::MemoryPool *pool) {
  switch (type->id()) {
    case arrow::Type::UINT8:return std::make_unique<UInt8ArrayStreamingSplitter>(targets, pool);
    case arrow::Type::INT8:return std::make_unique<Int8ArrayStreamingSplitter>(targets, pool);
    case arrow::Type::UINT16:return std::make_unique<UInt16ArrayStreamingSplitter>(targets, pool);
    case arrow::Type::INT16:return std::make_unique<Int16ArrayStreamingSplitter>(targets, pool);
    case arrow::Type::UINT32:return std::make_unique<UInt32ArrayStreamingSplitter>(targets, pool);
    case arrow::Type::INT32:return std::make_unique<Int32ArrayStreamingSplitter>(targets, pool);
    case arrow::Type::UINT64:return std::make_unique<UInt64ArrayStreamingSplitter>(targets, pool);
    case arrow::Type::INT64:return std::make_unique<Int64ArrayStreamingSplitter>(targets, pool);
    case arrow::Type::FLOAT:return std::make_unique<FloatArrayStreamingSplitter>(targets, pool);
    case arrow::Type::DOUBLE:return std::make_unique<DoubleArrayStreamingSplitter>(targets, pool);
    case arrow::Type::FIXED_SIZE_BINARY:return std::make_unique<FixedBinaryStreamingSplitKernel>(type, targets, pool);
    case arrow::Type::STRING:return std::make_unique<BinaryStreamingSplitKernel<arrow::StringType>>(targets, pool);
    case arrow::Type::BINARY:return std::make_unique<BinaryStreamingSplitKernel<arrow::BinaryType>>(targets, pool);
    default:LOG(FATAL) << "Un-known type " << type->name();
      return nullptr;
  }
}
}  // namespace cylon

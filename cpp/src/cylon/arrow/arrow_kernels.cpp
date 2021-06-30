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

#include "arrow_kernels.hpp"

#include <glog/logging.h>

#include <type_traits>

#include "../util/macros.hpp"
#include "../util/sort.hpp"
#include "../util/arrow_utils.hpp"
#include "arrow_comparator.hpp"

namespace cylon {

// SPLITTING -----------------------------------------------------------------------------

template <typename TYPE,
    typename = typename std::enable_if<arrow::is_number_type<TYPE>::value | arrow::is_boolean_type<TYPE>::value
                                           | arrow::is_temporal_type<TYPE>::value>::type>
class ArrowArrayNumericSplitKernel : public ArrowArraySplitKernel {
 public:
  explicit ArrowArrayNumericSplitKernel(arrow::MemoryPool *pool) : ArrowArraySplitKernel(pool) {}
  using T = typename TYPE::c_type;

  Status Split(const std::shared_ptr<arrow::ChunkedArray> &values, uint32_t num_partitions,
               const std::vector<uint32_t> &target_partitions, const std::vector<uint32_t> &counts,
               std::vector<std::shared_ptr<arrow::Array>> &output) override {
    if ((size_t)values->length() != target_partitions.size()) {
      return Status(Code::ExecutionError, "values rows != target_partitions length");
    }

    std::vector<std::shared_ptr<arrow::Buffer>> build_buffers;
    std::vector<T *> data_buffers;
    std::vector<int> buffer_indexes;
    for (uint32_t i = 0; i < num_partitions; i++) {
      int64_t buf_size = counts[i] * sizeof(T);
      arrow::Result<std::unique_ptr<arrow::Buffer>> result = arrow::AllocateBuffer(buf_size, pool_);
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
      std::shared_ptr<arrow::Buffer> indices_buf(std::move(result.ValueOrDie()));
      build_buffers.push_back(indices_buf);
      auto *indices_begin = reinterpret_cast<T *>(build_buffers.back()->mutable_data());
      data_buffers.push_back(indices_begin);
      buffer_indexes.push_back(0);
    }

    size_t offset = 0;
    for (const auto &array : values->chunks()) {
      const std::shared_ptr<arrow::ArrayData> &data = array->data();
      const T *value_buffer = data->template GetValues<T>(1);

      const int64_t arr_len = array->length();
      for (int64_t i = 0; i < arr_len; i++, offset++) {
        unsigned int target_buffer = target_partitions[offset];
        int idx = buffer_indexes[target_buffer];
        data_buffers[target_buffer][idx] = value_buffer[i];
        buffer_indexes[target_buffer] = idx + 1;
      }
    }

    output.reserve(num_partitions);
    for (uint32_t i = 0; i < num_partitions; i++) {
      std::vector<std::shared_ptr<arrow::Buffer>> buffs;
      buffs.push_back(nullptr);
      buffs.push_back(build_buffers[i]);
      const std::shared_ptr<arrow::ArrayData> &data = arrow::ArrayData::Make(values->type(), counts[i], buffs, 0, 0);
      std::shared_ptr<arrow::Array> array = arrow::MakeArray(data);
      output.push_back(array);
    }
    return Status::OK();
  }
};

class FixedBinaryArraySplitKernel : public ArrowArraySplitKernel {
 public:
  using ARROW_ARRAY_T = arrow::FixedSizeBinaryArray;
  using ARROW_BUILDER_T = arrow::FixedSizeBinaryBuilder;

  explicit FixedBinaryArraySplitKernel(arrow::MemoryPool *pool) : ArrowArraySplitKernel(pool) {}

  Status Split(const std::shared_ptr<arrow::ChunkedArray> &values, uint32_t num_partitions,
               const std::vector<uint32_t> &target_partitions, const std::vector<uint32_t> &counts,
               std::vector<std::shared_ptr<arrow::Array>> &output) override {
    if ((size_t)values->length() != target_partitions.size()) {
      return Status(Code::ExecutionError, "values rows != target_partitions length");
    }

    std::vector<std::unique_ptr<ARROW_BUILDER_T>> builders;
    builders.reserve(num_partitions);
    for (uint32_t i = 0; i < num_partitions; i++) {
      builders.push_back(std::make_unique<ARROW_BUILDER_T>(values->type(), pool_));
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(builders.back()->Reserve(counts[i]));
    }

    size_t offset = 0;
    for (const auto &array : values->chunks()) {
      std::shared_ptr<ARROW_ARRAY_T> casted_array = std::static_pointer_cast<ARROW_ARRAY_T>(array);
      const int64_t arr_len = array->length();
      for (int64_t i = 0; i < arr_len; i++, offset++) {
        RETURN_CYLON_STATUS_IF_ARROW_FAILED(builders[target_partitions[offset]]->Append(casted_array->Value(i)));
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

template <typename TYPE>
class BinaryArraySplitKernel : public ArrowArraySplitKernel {
 public:
  using ARROW_ARRAY_T = typename arrow::TypeTraits<TYPE>::ArrayType;
  using ARROW_BUILDER_T = typename arrow::TypeTraits<TYPE>::BuilderType;
  using ARROW_OFFSET_T = typename TYPE::offset_type;

  explicit BinaryArraySplitKernel(arrow::MemoryPool *pool) : ArrowArraySplitKernel(pool) {}

  Status Split(const std::shared_ptr<arrow::ChunkedArray> &values, uint32_t num_partitions,
               const std::vector<uint32_t> &target_partitions, const std::vector<uint32_t> &counts,
               std::vector<std::shared_ptr<arrow::Array>> &output) override {
    if ((size_t)values->length() != target_partitions.size()) {
      return Status(Code::ExecutionError, "values rows != target_partitions length");
    }

    std::vector<std::unique_ptr<ARROW_BUILDER_T>> builders;
    builders.reserve(num_partitions);
    for (uint32_t i = 0; i < num_partitions; i++) {
      builders.emplace_back(std::make_unique<ARROW_BUILDER_T>(pool_));
    }

    size_t offset = 0;
    for (const auto &array : values->chunks()) {
      std::shared_ptr<ARROW_ARRAY_T> casted_array = std::static_pointer_cast<ARROW_ARRAY_T>(array);
      const int64_t arr_len = array->length();
      for (int64_t i = 0; i < arr_len; i++, offset++) {
        ARROW_OFFSET_T length = 0;
        const uint8_t *value = casted_array->GetValue(i, &length);
        RETURN_CYLON_STATUS_IF_ARROW_FAILED(builders[target_partitions[offset]]->Append(value, length));
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

template <typename ARROW_T>
class ArrowBinarySortKernel : public IndexSortKernel {
  using ARRAY_T = typename arrow::TypeTraits<ARROW_T>::ArrayType;

 public:
  ArrowBinarySortKernel(arrow::MemoryPool *pool, bool ascending)
      : IndexSortKernel(pool, ascending) {}

  arrow::Status Sort(const std::shared_ptr<arrow::Array> &values,
                     std::shared_ptr<arrow::UInt64Array> &offsets) const override {
    auto array = std::static_pointer_cast<ARRAY_T>(values);

    if (ascending) {
      return DoSort(
          [&array](uint64_t left, uint64_t right) {
            return array->GetView(left).compare(array->GetView(right)) < 0;
          },
          values->length(), pool_, offsets);
    } else {
      return DoSort(
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
      return DoSort([&left_data](uint64_t left,
                                 uint64_t right) { return left_data[left] < left_data[right]; },
                    values->length(), pool_, offsets);
    } else {
      return DoSort([&left_data](uint64_t left,
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
    return arrow::Status(arrow::StatusCode::NotImplemented,
                         "unknown type " + values->type()->ToString());
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
    int64_t buf_size = length * sizeof(uint64_t);

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
    return arrow::Status(arrow::StatusCode::NotImplemented,
                         "unknown type " + values->type()->ToString());
  }
  return out->Sort(values, offsets);
}

arrow::Status IndexSortKernel::DoSort(const std::function<bool(int64_t, int64_t)> &comp,
                                      int64_t len, arrow::MemoryPool *pool,
                                      std::shared_ptr<arrow::UInt64Array> &offsets) {
  int64_t buf_size = len * sizeof(int64_t);

  arrow::Result<std::unique_ptr<arrow::Buffer>> result = arrow::AllocateBuffer(buf_size + 1, pool);
  const arrow::Status &status = result.status();
  if (!status.ok()) {
    LOG(FATAL) << "Failed to allocate sort indices - " << status.message();
    return status;
  }
  std::shared_ptr<arrow::Buffer> indices_buf(std::move(result.ValueOrDie()));

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
    return arrow::Status(arrow::StatusCode::Invalid,
                         "No of sort columns and no of sort direction indicators mismatch");
  }

  std::vector<std::shared_ptr<ArrayIndexComparator>> comparators;
  comparators.reserve(columns.size());
    for (size_t i = 0; i < columns.size(); i++) {
      comparators.push_back(
          CreateArrayIndexComparator(cylon::util::GetChunkOrEmptyArray(table->column(columns[i]), 0), ascending[i]));
    }

  int64_t buf_size = table->num_rows() * sizeof(int64_t);

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
    for (auto const &comp : comparators) {
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

template <typename TYPE>
class NumericStreamingSplitKernel : public StreamingSplitKernel {
  using ARRAY_T = typename arrow::TypeTraits<TYPE>::ArrayType;
  using T = typename TYPE::c_type;

public:
  NumericStreamingSplitKernel(int32_t num_targets, arrow::MemoryPool *pool)
      : StreamingSplitKernel(), num_partitions(num_targets), pool_(pool) {
    for (int i = 0; i < num_targets; i++) {
      int64_t buf_size = 1024 * 1024 * sizeof(T);
      arrow::Result<std::unique_ptr<arrow::ResizableBuffer>> result = arrow::AllocateResizableBuffer(buf_size, pool_);
      std::shared_ptr<arrow::ResizableBuffer> indices_buf(std::move(result.ValueOrDie()));
      build_buffers.push_back(indices_buf);

      counts.push_back(0);
      buffer_indexes.push_back(0);
    }
  }

  Status Split(const std::shared_ptr<arrow::Array> &values, const std::vector<uint32_t> &partitions,
               const std::vector<uint32_t> &cnts) override {
    const auto &cast_array = std::static_pointer_cast<ARRAY_T>(values);
    type = values->type();
    // reserve additional space in the builders
    data_buffers.clear();
    for (int i = 0; i < num_partitions; i++) {
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
    return Status::OK();
  }

  Status Finish(std::vector<std::shared_ptr<arrow::Array>> &out) override {
    out.reserve(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      std::vector<std::shared_ptr<arrow::Buffer>> buffs;
      buffs.push_back(nullptr);
      buffs.push_back(build_buffers[i]);
      const std::shared_ptr<arrow::ArrayData> &data = arrow::ArrayData::Make(type, counts[i], buffs);
      std::shared_ptr<arrow::Array> array = arrow::MakeArray(data);
      out.push_back(array);
    }
    return Status::OK();
  }

private:
  int32_t num_partitions;
  std::shared_ptr<arrow::DataType> type;
  std::vector<int> counts;
  std::vector<std::shared_ptr<arrow::ResizableBuffer>> build_buffers;
  std::vector<T *> data_buffers;
  std::vector<int> buffer_indexes;
  arrow::MemoryPool *pool_;
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

    for (size_t i = 0; i < partitions.size(); i++) {
      const uint8_t *value = reader->Value(i);
      builders_[partitions[i]]->UnsafeAppend(value);
    }
    return Status::OK();
  }

  Status Finish(std::vector<std::shared_ptr<arrow::Array>> &out) override {
    out.reserve(builders_.size());
    for (auto &builder : builders_) {
      std::shared_ptr<arrow::Array> array;
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder->Finish(&array));
      out.emplace_back(std::move(array));
    }
    return Status::OK();
  }

 private:
  std::vector<std::shared_ptr<arrow::FixedSizeBinaryBuilder>> builders_;
};

template <typename TYPE>
class BinaryStreamingSplitKernel : public StreamingSplitKernel {
  using BUILDER_T = typename arrow::TypeTraits<TYPE>::BuilderType;

 public:
  BinaryStreamingSplitKernel(int32_t &targets, arrow::MemoryPool *pool)
      : StreamingSplitKernel(), builders_({}) {
    builders_.reserve(targets);
    for (int i = 0; i < targets; i++) {
      builders_.emplace_back(std::make_shared<BUILDER_T>(pool));
    }
  }

  Status Split(const std::shared_ptr<arrow::Array> &values, const std::vector<uint32_t> &partitions,
               const std::vector<uint32_t> &cnts) override {
    auto reader = std::static_pointer_cast<arrow::BinaryArray>(values);

    for (size_t i = 0; i < partitions.size(); i++) {
      int length = 0;
      const uint8_t *value = reader->GetValue(i, &length);
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(builders_[partitions[i]]->Append(value, length));
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
    case arrow::Type::UINT8:
      return std::make_unique<UInt8ArrayStreamingSplitter>(targets, pool);
    case arrow::Type::INT8:
      return std::make_unique<Int8ArrayStreamingSplitter>(targets, pool);
    case arrow::Type::UINT16:
      return std::make_unique<UInt16ArrayStreamingSplitter>(targets, pool);
    case arrow::Type::INT16:
      return std::make_unique<Int16ArrayStreamingSplitter>(targets, pool);
    case arrow::Type::UINT32:
      return std::make_unique<UInt32ArrayStreamingSplitter>(targets, pool);
    case arrow::Type::INT32:
      return std::make_unique<Int32ArrayStreamingSplitter>(targets, pool);
    case arrow::Type::UINT64:
      return std::make_unique<UInt64ArrayStreamingSplitter>(targets, pool);
    case arrow::Type::INT64:
      return std::make_unique<Int64ArrayStreamingSplitter>(targets, pool);
    case arrow::Type::FLOAT:
      return std::make_unique<FloatArrayStreamingSplitter>(targets, pool);
    case arrow::Type::DOUBLE:
      return std::make_unique<DoubleArrayStreamingSplitter>(targets, pool);
    case arrow::Type::FIXED_SIZE_BINARY:
      return std::make_unique<FixedBinaryStreamingSplitKernel>(type, targets, pool);
    case arrow::Type::STRING:
      return std::make_unique<BinaryStreamingSplitKernel<arrow::StringType>>(targets, pool);
    case arrow::Type::BINARY:
      return std::make_unique<BinaryStreamingSplitKernel<arrow::BinaryType>>(targets, pool);
    default:
      LOG(FATAL) << "Un-known type " << type->name();
      return nullptr;
  }
}
}  // namespace cylon

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
#include "flatten_array.hpp"

#include <arrow/visit_data_inline.h>
#include <arrow/util/bitmap_visit.h>

#include "cylon/arrow/arrow_type_traits.hpp"

namespace cylon {

void IncrementRowNullCount(const arrow::ArrayData *arr_data, std::vector<int32_t> *row_null_count) {
  if (arr_data->buffers[0]) {
    // if validity bitmap present
    int64_t i = 0;
    auto bit_visitor = [&](bool bit) {
      // if bit == true --> valid, else invalid/null
      (*row_null_count)[i] += !bit;
      i++;
    };
    arrow::internal::VisitBitsUnrolled(arr_data->buffers[0]->data(),
                                       arr_data->offset,
                                       arr_data->length,
                                       bit_visitor);
  }
}

struct ColumnFlattenKernel {
  ColumnFlattenKernel() = default;
  virtual ~ColumnFlattenKernel() = default;
  virtual int32_t ByteWidth() const = 0;

  virtual Status CopyData(uint8_t col_idx,
                          int32_t *row_offset,
                          uint8_t *data_buf,
                          const int32_t *offset_buff) const = 0;

  virtual Status IncrementRowOffset(int32_t *offsets) const = 0;
};

// primitive types
template<typename ArrowT, typename Enable = arrow::enable_if_primitive_ctype<ArrowT>>
struct NumericFlattenKernelImpl : public ColumnFlattenKernel {
  using ValueT = typename ArrowTypeTraits<ArrowT>::ValueT;

  const arrow::ArrayData *array_data;
  explicit NumericFlattenKernelImpl(const arrow::ArrayData *array_data) : array_data(array_data) {}

  int32_t ByteWidth() const override {
    return sizeof(ValueT);
  }

  Status CopyData(uint8_t col_idx,
                  int32_t *row_offset,
                  uint8_t *data_buf,
                  const int32_t *offset_buff) const override {
    if (array_data->MayHaveNulls()) {
      int64_t i = 0;
      arrow::VisitArraySpanInline<ArrowT>(*array_data,
                                          [&](const ValueT &val) {
                                            std::memcpy(data_buf + offset_buff[i] + row_offset[i],
                                                        &val, sizeof(ValueT));
                                            row_offset[i] += sizeof(ValueT);
                                            assert(row_offset[i] <= offset_buff[i + 1]);
                                            i++;
                                          },
                                          [&]() {
                                            // clear the slot
                                            std::memset(data_buf + offset_buff[i] + row_offset[i],
                                                        0, sizeof(ValueT));
                                            row_offset[i] += sizeof(ValueT);
                                            assert(row_offset[i] <= offset_buff[i + 1]);

                                            // now copy the col_idx to appropriate place
                                            uint8_t curr_row_null_count = data_buf[offset_buff[i]];
                                            data_buf[offset_buff[i] + 1 + curr_row_null_count] =
                                                col_idx;
                                            data_buf[offset_buff[i]]++; // increment curr_row_null_count
                                            i++;
                                          });
    } else {
      const auto *data = array_data->template GetValues<ValueT>(1);
      for (int64_t i = 0; i < array_data->length; i++) {
        std::memcpy(data_buf + offset_buff[i] + row_offset[i], data + i, sizeof(ValueT));
        row_offset[i] += sizeof(ValueT);
        assert(row_offset[i] <= offset_buff[i + 1]);
      }
    }
    return Status::OK();
  }

  Status IncrementRowOffset(int32_t *offsets) const override {
    CYLON_UNUSED(offsets);
    return {Code::Invalid, "Do not individually increment row-offsets for primitive arrays"};
  }
};

// binary types 
template<typename ArrowT, typename Enable = arrow::enable_if_base_binary<ArrowT>>
struct BinaryColumnFlattenKernelImpl : public ColumnFlattenKernel {
  using ValueT = typename ArrowTypeTraits<ArrowT>::ValueT;
  using OffsetType = typename arrow::TypeTraits<ArrowT>::OffsetType::c_type;

  const arrow::ArrayData *array_data;
  explicit BinaryColumnFlattenKernelImpl(const arrow::ArrayData *array_data)
      : array_data(array_data) {}

  int32_t ByteWidth() const override {
    return -1;
  }

  Status CopyData(uint8_t col_idx,
                  int32_t *row_offset,
                  uint8_t *data_buf,
                  const int32_t *offset_buff) const override {
    if (array_data->MayHaveNulls()) {
      int64_t i = 0;
      arrow::VisitArraySpanInline<ArrowT>(*array_data,
                                          [&](const ValueT &val) {
                                            std::memcpy(data_buf + offset_buff[i] + row_offset[i],
                                                        val.data(), val.size());
                                            row_offset[i] += val.size();
                                            assert(row_offset[i] <= offset_buff[i + 1]);
                                            i++;
                                          },
                                          [&]() {
                                            // nothing to copy to data buffer

                                            // now copy the col_idx to appropriate place
                                            uint8_t curr_row_null_count = data_buf[offset_buff[i]];
                                            data_buf[offset_buff[i] + 1 + curr_row_null_count] =
                                                col_idx;
                                            data_buf[offset_buff[i]]++; // increment curr_row_null_count
                                            i++;
                                          });
    } else {
      const auto *offsets = array_data->template GetValues<const OffsetType>(1);
      const auto *data = array_data->buffers[2]->data();
      for (int64_t i = 0; i < array_data->length; i++) {
        OffsetType size = offsets[i + 1] - offsets[i];
        std::memcpy(data_buf + offset_buff[i] + row_offset[i], data + offsets[i], size);
        row_offset[i] += size;
        assert(row_offset[i] <= offset_buff[i + 1]);
      }
    }
    return Status::OK();
  }

  Status IncrementRowOffset(int32_t *offsets) const override {
    if (!array_data->MayHaveNulls()) {
      const auto *arr_offsets = array_data->template GetValues<OffsetType>(1); // get offsets
      for (int64_t i = 1; i < array_data->length + 1; i++) { // dont update offsets[0]
        offsets[i] += (arr_offsets[i] - arr_offsets[i - 1]);
      }
    } else {
      int64_t i = 1; // dont update offsets[0]
      arrow::VisitArraySpanInline<ArrowT>(*array_data,
                                          [&](const std::string_view &val) {
                                            offsets[i] += static_cast<int32_t>(val.size());
                                            i++;
                                          },
                                          [&]() {
                                            i++;
                                          });
    }
    return Status::OK();
  }
};

std::unique_ptr<ColumnFlattenKernel> GetKernel(const std::shared_ptr<arrow::Array> &array) {
  switch (array->type_id()) {
    case arrow::Type::BOOL:
      return std::make_unique<NumericFlattenKernelImpl<arrow::BooleanType>>(array->data().get());
    case arrow::Type::UINT8:
    case arrow::Type::INT8:
      return std::make_unique<NumericFlattenKernelImpl<arrow::UInt8Type>>(array->data().get());
    case arrow::Type::UINT16:
    case arrow::Type::INT16:
      return std::make_unique<NumericFlattenKernelImpl<arrow::UInt16Type>>(array->data().get());
    case arrow::Type::UINT32:
    case arrow::Type::INT32:
    case arrow::Type::FLOAT:
    case arrow::Type::DATE32:
    case arrow::Type::TIME32:
    case arrow::Type::INTERVAL_MONTHS:
      return std::make_unique<NumericFlattenKernelImpl<arrow::UInt32Type>>(array->data().get());
    case arrow::Type::UINT64:
    case arrow::Type::INT64:
    case arrow::Type::DOUBLE:
    case arrow::Type::DATE64:
    case arrow::Type::TIMESTAMP:
    case arrow::Type::TIME64:
    case arrow::Type::DURATION:
    case arrow::Type::INTERVAL_DAY_TIME:
      return std::make_unique<NumericFlattenKernelImpl<arrow::UInt64Type>>(array->data().get());
    case arrow::Type::STRING:
    case arrow::Type::BINARY:
      return std::make_unique<BinaryColumnFlattenKernelImpl<arrow::BinaryType>>(array->data()
                                                                                    .get());
    case arrow::Type::LARGE_STRING:
    case arrow::Type::LARGE_BINARY:
      return std::make_unique<BinaryColumnFlattenKernelImpl<arrow::LargeBinaryType>>(array->data()
                                                                                         .get());
    case arrow::Type::NA:
    case arrow::Type::HALF_FLOAT:
    case arrow::Type::FIXED_SIZE_BINARY:
    case arrow::Type::DECIMAL128:
    case arrow::Type::DECIMAL256:
    case arrow::Type::LIST:
    case arrow::Type::STRUCT:
    case arrow::Type::SPARSE_UNION:
    case arrow::Type::DENSE_UNION:
    case arrow::Type::DICTIONARY:
    case arrow::Type::MAP:
    case arrow::Type::EXTENSION:
    case arrow::Type::FIXED_SIZE_LIST:
    case arrow::Type::LARGE_LIST:
    case arrow::Type::MAX_ID:
    default:return nullptr;
  }
}

// 1 byte to hold how many number of nulls in a row
static constexpr int32_t additional_data = sizeof(uint8_t);

/**
 * Calculates row offsets as follows.
 * if any array contains nulls,
 *  then at i'th row, offsets[i] = 1 + num_nulls_per_row + flattened_row_data_size (1 byte to track num_nulls)
 *  else, offsets[i] = flattened_row_data_size
 * @param flatten_kernels
 * @param metadata
 * @param row_offset
 * @param len
 * @param offsets
 * @return
 */
Status CalculateRowOffsets(const std::vector<std::unique_ptr<ColumnFlattenKernel>> &flatten_kernels,
                           const ArraysMetadata &metadata,
                           const std::vector<int32_t> &row_offset,
                           int64_t len,
                           int32_t *offsets) {
  assert(metadata.fixed_size_bytes_per_row || !metadata.var_bin_array_indices.empty());

  if ((additional_data + metadata.fixed_size_bytes_per_row) * len > INT32_MAX) {
    return {Code::NotImplemented, "Flatten arrays cannot handle offsets > INT32_MAX"};
  }

  // initialize offset array with row_offset
  offsets[0] = 0; // mutable_offset[0] will be skipped
  if (metadata.ContainsNullArrays()) {
    // if there are nulls, reserve an additional space (1 byte) for num nulls
    std::transform(row_offset.cbegin(),
                   row_offset.cend(),
                   offsets + 1,
                   [&](const int32_t row_nulls) {
                     return row_nulls + metadata.fixed_size_bytes_per_row;
                   });
  } else { // all columns are valid. So no need to reserve additional byte
    // add metadata.fixed_size_bytes_per_row to each position of the offset
    std::fill(offsets + 1, offsets + len + 1, metadata.fixed_size_bytes_per_row);
  }

  // if there are no variable binary arrays, adjustment for fixed_size_bytes_per_row and metadata is already done!
  if (metadata.ContainsOnlyNumeric()) {
    return Status::OK();
  }

  // traverse through the var_bin_arrays
  for (const auto &idx: metadata.var_bin_array_indices) {
    const auto *kernel = flatten_kernels[idx].get();
    RETURN_CYLON_STATUS_IF_FAILED(kernel->IncrementRowOffset(offsets));
  }

  return Status::OK();
}

Status FlattenArrays(CylonContext *ctx, const std::vector<std::shared_ptr<arrow::Array>> &arrays,
                     std::shared_ptr<FlattenedArray> *output) {
  if (arrays.size() > 256) {
    return {Code::NotImplemented, "FlattenArrays only support maximum 256 arrays"};
  }

  if (arrays.empty()) {
    return Status::OK();
  }

  auto pool = ToArrowPool(ctx);

  const int64_t len = arrays[0]->length();
  if (std::any_of(arrays.begin() + 1, arrays.end(), [&](const std::shared_ptr<arrow::Array> &arr) {
    return arr->length() != len;
  })) {
    return {Code::Invalid, "array lengths should be the same"};
  }

  // traverse the arrays and create metadata
  ArraysMetadata metadata;
  std::vector<std::unique_ptr<ColumnFlattenKernel>> flatten_kernels;
  flatten_kernels.reserve(arrays.size());

  // row offsets (this vector serves 2 purposes. 1. count nulls per row, 2. carry row offsets)
  // this vector will be lazily initialized
  std::vector<int32_t> row_offsets;
  // if at least one array has nulls, init to 1, else init to 0
  if (std::any_of(arrays.begin(), arrays.end(), [&](const std::shared_ptr<arrow::Array> &arr) {
    return arr->null_count() > 0;
  })) {
    row_offsets.resize(len, additional_data);
  } else {
    row_offsets.resize(len, 0);
  }

  for (uint8_t i = 0; i < static_cast<uint8_t>(arrays.size()); i++) {
    const auto &arr = arrays[i];
    auto kernel = GetKernel(arr);
    if (kernel == nullptr) {
      return {Code::NotImplemented, "unsupported type " + arr->type()->ToString()};
    }
    flatten_kernels.push_back(std::move(kernel));

    bool has_nulls = arr->null_count() > 0;
    if (has_nulls) {
      IncrementRowNullCount(arr->data().get(), &row_offsets);
      metadata.arrays_with_nulls++;
    }

    int byte_width = flatten_kernels.back()->ByteWidth();
    if (byte_width >= 0) {
      // this is a fixed sized type.
      metadata.fixed_size_bytes_per_row += byte_width;
    } else {
      metadata.var_bin_array_indices.push_back(i);
    }
  }
  // at this point, row_offsets contains num nulls at each row

  // create the offset array
  CYLON_ASSIGN_OR_RAISE(auto offset_buf, arrow::AllocateBuffer((len + 1) * sizeof(int32_t), pool))
  auto *offsets = reinterpret_cast<int32_t *>(offset_buf->mutable_data());
  std::fill(offsets, offsets + len + 1, 0);

  // create offset sizes
  RETURN_CYLON_STATUS_IF_FAILED(CalculateRowOffsets(flatten_kernels,
                                                    metadata,
                                                    row_offsets,
                                                    len,
                                                    offsets));
  // convert to prefix sum
  std::partial_sum(offsets, offsets + len + 1, offsets);
  int32_t total_size = offsets[len];

  // now allocate data array
  CYLON_ASSIGN_OR_RAISE(auto data_buf, arrow::AllocateBuffer(total_size, pool))
  // initialize num nulls
  if (metadata.ContainsNullArrays()) {
    auto *data = data_buf->mutable_data();
    std::for_each(offsets, offsets + len, [&](int32_t offset) { data[offset] = 0; });
  }

  // now copy the data
  for (uint8_t i = 0; i < static_cast<uint8_t>(flatten_kernels.size()); i++) {
    const ColumnFlattenKernel *kernel = flatten_kernels[i].get();
    RETURN_CYLON_STATUS_IF_FAILED(kernel->CopyData(i,
                                                   row_offsets.data(),
                                                   data_buf->mutable_data(),
                                                   offsets));
  }

  auto flattened = std::make_shared<arrow::BinaryArray>(len,
                                                        std::move(offset_buf),
                                                        std::move(data_buf),
                                                        nullptr,
                                                        0);

  *output = std::make_shared<FlattenedArray>(std::move(flattened), arrays, std::move(metadata));

  return Status::OK();
}

}





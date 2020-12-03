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

#include <arrow/api.h>
#include <glog/logging.h>

#include <vector>
#include <memory>
#include <arrow/arrow_kernels.hpp>
#include <random>

#include "arrow_utils.hpp"

namespace cylon {
namespace util {

template<typename TYPE>
arrow::Status SortNumericColumn(const std::shared_ptr<arrow::Array> &data_column,
                                const std::shared_ptr<arrow::Int64Array> &sorted_indices,
                                std::shared_ptr<arrow::Array> *sorted_array,
                                arrow::MemoryPool *memory_pool) {
  const int64_t length = sorted_indices->length();

  using ARROW_BUILDER_T = typename arrow::TypeTraits<TYPE>::BuilderType;
  using ARROW_ARRAY_T = typename arrow::TypeTraits<TYPE>::ArrayType;

  ARROW_BUILDER_T array_builder(memory_pool);
  arrow::Status reserveStatus = array_builder.Reserve(length);

  auto data_array_values = std::static_pointer_cast<ARROW_ARRAY_T>(data_column);
  auto sorted_idx_values = sorted_indices->raw_values();
  for (int64_t index = 0; index < length; ++index) {
    int64_t current_index = sorted_idx_values[index];
    array_builder.UnsafeAppend(data_array_values->Value(current_index));
  }
  return array_builder.Finish(sorted_array);
}

arrow::Status SortColumn(const std::shared_ptr<arrow::Array> &data_column,
                         const std::shared_ptr<arrow::Int64Array> &sorted_indices,
                         std::shared_ptr<arrow::Array> *sorted_column_array,
                         arrow::MemoryPool *memory_pool) {
  // todo support non numeric types
  switch (data_column->type()->id()) {
    case arrow::Type::UINT8:
      return SortNumericColumn<arrow::UInt8Type>(data_column,
                                                 sorted_indices,
                                                 sorted_column_array,
                                                 memory_pool);
    case arrow::Type::NA:break;
    case arrow::Type::BOOL:
      return SortNumericColumn<arrow::BooleanType>(data_column,
                                                   sorted_indices,
                                                   sorted_column_array,
                                                   memory_pool);
    case arrow::Type::INT8:
      return SortNumericColumn<arrow::Int8Type>(data_column,
                                                sorted_indices,
                                                sorted_column_array,
                                                memory_pool);
    case arrow::Type::UINT16:
      return SortNumericColumn<arrow::UInt16Type>(data_column,
                                                  sorted_indices,
                                                  sorted_column_array,
                                                  memory_pool);
    case arrow::Type::INT16:
      return SortNumericColumn<arrow::Int16Type>(data_column,
                                                 sorted_indices,
                                                 sorted_column_array,
                                                 memory_pool);
    case arrow::Type::UINT32:
      return SortNumericColumn<arrow::UInt32Type>(data_column,
                                                  sorted_indices,
                                                  sorted_column_array,
                                                  memory_pool);
    case arrow::Type::INT32:
      return SortNumericColumn<arrow::Int32Type>(data_column,
                                                 sorted_indices,
                                                 sorted_column_array,
                                                 memory_pool);
    case arrow::Type::UINT64:
      return SortNumericColumn<arrow::UInt64Type>(data_column,
                                                  sorted_indices,
                                                  sorted_column_array,
                                                  memory_pool);
    case arrow::Type::INT64:
      return SortNumericColumn<arrow::Int64Type>(data_column,
                                                 sorted_indices,
                                                 sorted_column_array,
                                                 memory_pool);
    case arrow::Type::HALF_FLOAT:
      return SortNumericColumn<arrow::HalfFloatType>(data_column,
                                                     sorted_indices,
                                                     sorted_column_array,
                                                     memory_pool);
    case arrow::Type::FLOAT:
      return SortNumericColumn<arrow::FloatType>(data_column,
                                                 sorted_indices,
                                                 sorted_column_array,
                                                 memory_pool);
    case arrow::Type::DOUBLE:
      return SortNumericColumn<arrow::DoubleType>(data_column,
                                                  sorted_indices,
                                                  sorted_column_array,
                                                  memory_pool);
    case arrow::Type::STRING:break;
    case arrow::Type::BINARY:break;
    case arrow::Type::FIXED_SIZE_BINARY:break;
    case arrow::Type::DATE32:break;
    case arrow::Type::DATE64:break;
    case arrow::Type::TIMESTAMP:break;
    case arrow::Type::TIME32:break;
    case arrow::Type::TIME64:break;
    case arrow::Type::DECIMAL:break;
    case arrow::Type::LIST:break;
    case arrow::Type::STRUCT:break;
    case arrow::Type::DICTIONARY:break;
    case arrow::Type::MAP:break;
    case arrow::Type::EXTENSION:break;
    case arrow::Type::FIXED_SIZE_LIST:break;
    case arrow::Type::DURATION:break;
    case arrow::Type::LARGE_STRING:break;
    case arrow::Type::LARGE_BINARY:break;
    case arrow::Type::LARGE_LIST:break;
    case arrow::Type::INTERVAL_MONTHS:break;
    case arrow::Type::INTERVAL_DAY_TIME:break;
    case arrow::Type::SPARSE_UNION:break;
    case arrow::Type::DENSE_UNION:break;
    case arrow::Type::MAX_ID:break;
  }
  return arrow::Status::OK();
}

arrow::Status SortTable(const std::shared_ptr<arrow::Table> &table, int64_t sort_column_index,
                        std::shared_ptr<arrow::Table> *sorted_table, arrow::MemoryPool *memory_pool) {
  std::shared_ptr<arrow::Table> tab_to_process; // table referenced
  // combine chunks if multiple chunks are available
  if (table->column(sort_column_index)->num_chunks() > 1) {
    arrow::Result<std::shared_ptr<arrow::Table>> left_combine_res = table->CombineChunks(memory_pool);
    if(!left_combine_res.ok()){
      return left_combine_res.status();
    }
    tab_to_process = left_combine_res.ValueOrDie();
  } else {
    tab_to_process = table;
  }
  auto column_to_sort = tab_to_process->column(sort_column_index)->chunk(0);

  // sort to indices
  std::shared_ptr<arrow::Array> sorted_column_index;
  arrow::Status status = cylon::SortIndices(memory_pool, column_to_sort, &sorted_column_index);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to sort column to indices" << status.ToString();
    return status;
  }

  auto index_lookup = std::static_pointer_cast<arrow::Int64Array>(sorted_column_index);

  // now sort everything based on sorted index
  arrow::ArrayVector sorted_columns;
  const int64_t no_of_columns = tab_to_process->num_columns();
  for (int64_t col_index = 0; col_index < no_of_columns; ++col_index) {
    std::shared_ptr<arrow::Array> sorted_array;
    status = SortColumn(tab_to_process->column(col_index)->chunk(0),
                        index_lookup, &sorted_array, memory_pool);
    if (!status.ok()) {
      LOG(FATAL) << "Failed to sort column based on indices. " << status.ToString();
      return status;
    }
    sorted_columns.push_back(sorted_array);
  }
  *sorted_table = arrow::Table::Make(table->schema(), sorted_columns);
  return arrow::Status::OK();
}

arrow::Status free_table(const std::shared_ptr<arrow::Table> &table) {
  const int ncolumns = table->num_columns();
  for (int i = 0; i < ncolumns; ++i) {
    auto col = table->column(i);
    int nChunks = col->num_chunks();
    for (int c = 0; c < nChunks; c++) {
      auto chunk = col->chunk(c);
      std::shared_ptr<arrow::ArrayData> ptr = chunk->data();
      for (const auto &t : ptr->buffers) {
        delete[] t->data();
      }
    }
  }
  return arrow::Status::OK();
}

arrow::Status duplicate(const std::shared_ptr<arrow::ChunkedArray>& cArr,
    const std::shared_ptr<arrow::Field>& field, arrow::MemoryPool *pool,
                        std::shared_ptr<arrow::ChunkedArray>& out) {
  size_t size = cArr->chunks().size();
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  for (size_t arrayIndex = 0; arrayIndex < size; arrayIndex++) {
    std::shared_ptr<arrow::Array> arr = cArr->chunk(arrayIndex);
    std::shared_ptr<arrow::ArrayData> data = arr->data();
    std::vector<std::shared_ptr<arrow::Buffer>> buffers;
    size_t length = cArr->length();
    for (const auto& buf : data->buffers) {
      arrow::Result<std::shared_ptr<arrow::Buffer>> res = buf->CopySlice(0l, buf->size(), pool);

      if (!res.ok()) {
        LOG(FATAL) << "Insufficient memory";
        return res.status();
      }
      buffers.push_back(res.ValueOrDie());
    }
    // lets send this buffer, we need to send the length at this point
    std::shared_ptr<arrow::ArrayData> new_data = arrow::ArrayData::Make(
        field->type(), length, buffers);
    std::shared_ptr<arrow::Array> array = arrow::MakeArray(data);
    arrays.push_back(array);
  }
  out = std::make_shared<arrow::ChunkedArray>(arrays, field->type());
  return arrow::Status::OK();
}

template<typename TYPE>
static arrow::Status sample_array(const std::shared_ptr<arrow::ChunkedArray> &array,
                                  uint64_t num_samples,
                                  std::shared_ptr<arrow::Array> &out) {
  using ARROW_BUILDER_T = typename arrow::TypeTraits<TYPE>::BuilderType;
  using ARROW_ARRAY_T = typename arrow::TypeTraits<TYPE>::ArrayType;

  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<int64_t> distrib(0, array->length() - 1);

  std::shared_ptr<arrow::Array> concat_arr;
  if (array->num_chunks() > 1) {
    const arrow::Result<std::shared_ptr<arrow::Array>> &res = arrow::Concatenate(array->chunks());
    RETURN_ARROW_STATUS_IF_FAILED(res.status())
    concat_arr = res.ValueOrDie();
  } else {
    concat_arr = array->chunk(0);
  }

  ARROW_BUILDER_T builder;
  auto a_status = builder.Reserve(num_samples);
  RETURN_ARROW_STATUS_IF_FAILED(a_status)

  auto casted_array = std::static_pointer_cast<ARROW_ARRAY_T>(concat_arr);
  for (uint64_t i = 0; i < num_samples; i++) {
    int64_t idx = distrib(gen);
    builder.UnsafeAppend(casted_array->Value(idx));
  }

  return builder.Finish(&out);
}

arrow::Status SampleTable(std::shared_ptr<arrow::Table> &table,
                          int32_t idx,
                          uint64_t num_samples,
                          std::shared_ptr<arrow::Array> &out) {
  return SampleArray(table->column(idx), num_samples, out);
}

arrow::Status SampleArray(const std::shared_ptr<arrow::ChunkedArray> &arr,
                          uint64_t num_samples,
                          std::shared_ptr<arrow::Array> &out) {
  switch (arr->type()->id()) {
    case arrow::Type::BOOL: return sample_array<arrow::BooleanType>(arr, num_samples, out);
    case arrow::Type::UINT8:return sample_array<arrow::UInt8Type>(arr, num_samples, out);
    case arrow::Type::INT8:return sample_array<arrow::Int8Type>(arr, num_samples, out);
    case arrow::Type::UINT16:return sample_array<arrow::UInt16Type>(arr, num_samples, out);
    case arrow::Type::INT16:return sample_array<arrow::Int16Type>(arr, num_samples, out);
    case arrow::Type::UINT32:return sample_array<arrow::UInt32Type>(arr, num_samples, out);
    case arrow::Type::INT32:return sample_array<arrow::Int32Type>(arr, num_samples, out);
    case arrow::Type::UINT64:return sample_array<arrow::UInt32Type>(arr, num_samples, out);
    case arrow::Type::INT64:return sample_array<arrow::Int64Type>(arr, num_samples, out);
    case arrow::Type::FLOAT:return sample_array<arrow::FloatType>(arr, num_samples, out);
    case arrow::Type::DOUBLE:return sample_array<arrow::DoubleType>(arr, num_samples, out);
    default: return arrow::Status(arrow::StatusCode::Invalid, "unsupported type");
  }
}

}  // namespace util
}  // namespace cylon

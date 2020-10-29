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

#include <arrow/compute/api.h>
#include <arrow/api.h>
#include <glog/logging.h>

#include <vector>
#include <memory>
#include <arrow/arrow_kernels.hpp>

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
    case arrow::Type::INTERVAL:break;
    case arrow::Type::DECIMAL:break;
    case arrow::Type::LIST:break;
    case arrow::Type::STRUCT:break;
    case arrow::Type::UNION:break;
    case arrow::Type::DICTIONARY:break;
    case arrow::Type::MAP:break;
    case arrow::Type::EXTENSION:break;
    case arrow::Type::FIXED_SIZE_LIST:break;
    case arrow::Type::DURATION:break;
    case arrow::Type::LARGE_STRING:break;
    case arrow::Type::LARGE_BINARY:break;
    case arrow::Type::LARGE_LIST:break;
  }
  return arrow::Status::OK();
}

arrow::Status SortTable(const std::shared_ptr<arrow::Table> &table, int64_t sort_column_index,
                        std::shared_ptr<arrow::Table> *sorted_table, arrow::MemoryPool *memory_pool) {
  std::shared_ptr<arrow::Table> tab_to_process; // table referenced
  // combine chunks if multiple chunks are available
  if (table->column(sort_column_index)->num_chunks() > 1) {
    arrow::Status left_combine_stat = table->CombineChunks(memory_pool, &tab_to_process);
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
    for (size_t bufferIndex = 0; bufferIndex < data->buffers.size(); bufferIndex++) {
      std::shared_ptr<arrow::Buffer> buf = data->buffers[bufferIndex];
      std::shared_ptr<arrow::Buffer> new_buf;
      arrow::Status st = buf->Copy(0l, buf->size(), pool, &new_buf);
      if (!st.ok()) {
        LOG(FATAL) << "Insufficient memory";
        return st;
      }
      buffers.push_back(new_buf);
    }
    // create the array with the new buffers
    std::shared_ptr<arrow::ArrayData> new_data = arrow::ArrayData::Make(
        field->type(), length, buffers);
    std::shared_ptr<arrow::Array> array = arrow::MakeArray(new_data);
    arrays.push_back(array);
  }
  out = std::make_shared<arrow::ChunkedArray>(arrays, field->type());
  return arrow::Status::OK();
}

}  // namespace util
}  // namespace cylon

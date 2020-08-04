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

#include "arrow_utils.hpp"

namespace cylon {
namespace util {

template<typename TYPE>
arrow::Status sort_numeric_column_type(const std::shared_ptr<arrow::Array> &data_column,
                                       const std::shared_ptr<arrow::Int64Array> &sorted_indices,
                                       std::shared_ptr<arrow::Array> *sorted_array,
                                       arrow::MemoryPool *memory_pool) {
  int64_t length = sorted_indices->length();
  arrow::NumericBuilder<TYPE> array_builder(memory_pool);
  arrow::Status reserveStatus = array_builder.Reserve(length);
  auto casted_data_array = std::static_pointer_cast<arrow::NumericArray<TYPE>>(data_column);
  for (int64_t index = 0; index < length; ++index) {
    int64_t current_index = sorted_indices->Value(index);
    arrow::Status status = array_builder.Append(casted_data_array->Value(current_index));
    if (status != arrow::Status::OK()) {
      LOG(FATAL) << "Failed to append new elements to the builder while sorting. "
                 << status.ToString();
      return status;
    }
  }
  return array_builder.Finish(sorted_array);
}

arrow::Status sort_column(const std::shared_ptr<arrow::Array> &data_column,
                          const std::shared_ptr<arrow::Int64Array> &sorted_indices,
                          std::shared_ptr<arrow::Array> *sorted_column_array,
                          arrow::MemoryPool *memory_pool) {
  // todo support non numeric types
  switch (data_column->type()->id()) {
    case arrow::Type::UINT8:
      return sort_numeric_column_type<arrow::UInt8Type>(data_column,
                                                        sorted_indices,
                                                        sorted_column_array,
                                                        memory_pool);
    case arrow::Type::NA:break;
    case arrow::Type::BOOL:
      return sort_numeric_column_type<arrow::BooleanType>(data_column,
                                                          sorted_indices,
                                                          sorted_column_array,
                                                          memory_pool);
    case arrow::Type::INT8:
      return sort_numeric_column_type<arrow::Int8Type>(data_column,
                                                       sorted_indices,
                                                       sorted_column_array,
                                                       memory_pool);
    case arrow::Type::UINT16:
      return sort_numeric_column_type<arrow::UInt16Type>(data_column,
                                                         sorted_indices,
                                                         sorted_column_array,
                                                         memory_pool);
    case arrow::Type::INT16:
      return sort_numeric_column_type<arrow::Int16Type>(data_column,
                                                        sorted_indices,
                                                        sorted_column_array,
                                                        memory_pool);
    case arrow::Type::UINT32:
      return sort_numeric_column_type<arrow::UInt32Type>(data_column,
                                                         sorted_indices,
                                                         sorted_column_array,
                                                         memory_pool);
    case arrow::Type::INT32:
      return sort_numeric_column_type<arrow::Int32Type>(data_column,
                                                        sorted_indices,
                                                        sorted_column_array,
                                                        memory_pool);
    case arrow::Type::UINT64:
      return sort_numeric_column_type<arrow::UInt64Type>(data_column,
                                                         sorted_indices,
                                                         sorted_column_array,
                                                         memory_pool);
    case arrow::Type::INT64:
      return sort_numeric_column_type<arrow::Int64Type>(data_column,
                                                        sorted_indices,
                                                        sorted_column_array,
                                                        memory_pool);
    case arrow::Type::HALF_FLOAT:
      return sort_numeric_column_type<arrow::HalfFloatType>(data_column,
                                                            sorted_indices,
                                                            sorted_column_array,
                                                            memory_pool);
    case arrow::Type::FLOAT:
      return sort_numeric_column_type<arrow::FloatType>(data_column,
                                                        sorted_indices,
                                                        sorted_column_array,
                                                        memory_pool);
    case arrow::Type::DOUBLE:
      return sort_numeric_column_type<arrow::DoubleType>(data_column,
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

arrow::Status sort_table(std::shared_ptr<arrow::Table> tab, int64_t sort_column_index,
                         std::shared_ptr<arrow::Table> *sorted_table,
                         arrow::MemoryPool *memory_pool) {
  std::shared_ptr<arrow::Table> tab_to_process;
  // combine chunks if multiple chunks are available
  if (tab->column(sort_column_index)->num_chunks() > 1) {
    arrow::Status left_combine_stat = tab->CombineChunks(memory_pool, &tab);
  } else {
    tab_to_process = tab;
  }
  auto column_to_sort = tab_to_process->column(sort_column_index)->chunk(0);

  // sort to indices
  std::shared_ptr<arrow::Array> sorted_column_index;
  arrow::compute::FunctionContext ctx;
  arrow::Status status = arrow::compute::SortToIndices(&ctx, *column_to_sort, &sorted_column_index);

  if (status != arrow::Status::OK()) {
    LOG(FATAL) << "Failed to sort column to indices" << status.ToString();
    return status;
  }

  auto index_lookup = std::static_pointer_cast<arrow::Int64Array>(sorted_column_index);

  // now sort everything based on sorted index
  std::vector<std::shared_ptr<arrow::Array>> sorted_columns;
  int64_t no_of_columns = tab_to_process->num_columns();
  for (int64_t col_index = 0; col_index < no_of_columns; ++col_index) {
    std::shared_ptr<arrow::Array> sorted_array;
    status = sort_column(tab_to_process->column(col_index)->chunk(0),
                         index_lookup, &sorted_array, memory_pool);
    if (status != arrow::Status::OK()) {
      LOG(FATAL) << "Failed to sort column based on indices. " << status.ToString();
      return status;
    }
    sorted_columns.push_back(sorted_array);
  }
  *sorted_table = arrow::Table::Make(tab->schema(), sorted_columns);
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

}  // namespace util
}  // namespace cylon

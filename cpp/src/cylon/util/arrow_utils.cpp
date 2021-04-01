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

#include "arrow_utils.hpp"

#include <arrow/api.h>
#include <arrow/compute/api.h>

#include <arrow/arrow_kernels.hpp>
#include <memory>
#include <random>
#include <vector>

#include "macros.hpp"

namespace cylon {
namespace util {

arrow::Status SortTable(const std::shared_ptr<arrow::Table> &table, int64_t sort_column_index,
                        arrow::MemoryPool *memory_pool, std::shared_ptr<arrow::Table> &sorted_table,
                        bool ascending) {
  std::shared_ptr<arrow::Table> tab_to_process;  // table referenced
  // combine chunks if multiple chunks are available
  if (table->column(sort_column_index)->num_chunks() > 1) {
    const auto &res = table->CombineChunks(memory_pool);
    RETURN_ARROW_STATUS_IF_FAILED(res.status());
    tab_to_process = res.ValueOrDie();
  } else {
    tab_to_process = table;
  }
  const std::shared_ptr<arrow::Array> &column_to_sort =
      cylon::util::GetChunkOrEmptyArray(tab_to_process->column(sort_column_index), 0);

  // sort to indices
  std::shared_ptr<arrow::UInt64Array> sorted_column_index;
  RETURN_ARROW_STATUS_IF_FAILED(
      cylon::SortIndices(memory_pool, column_to_sort, sorted_column_index, ascending));

  // now sort everything based on sorted index
  arrow::ArrayVector sorted_columns;
  sorted_columns.reserve(table->num_columns());

  arrow::compute::ExecContext exec_context(memory_pool);
  // no bounds check is needed as indices are guaranteed to be within range
  const arrow::compute::TakeOptions &take_options = arrow::compute::TakeOptions::NoBoundsCheck();

  for (int64_t col_index = 0; col_index < tab_to_process->num_columns(); ++col_index) {
    const arrow::Result<arrow::Datum> &res = arrow::compute::Take(
        cylon::util::GetChunkOrEmptyArray(tab_to_process->column(col_index), 0),
        sorted_column_index, take_options, &exec_context);
    RETURN_ARROW_STATUS_IF_FAILED(res.status());
    sorted_columns.emplace_back(res.ValueOrDie().make_array());
  }

  sorted_table = arrow::Table::Make(table->schema(), sorted_columns);
  return arrow::Status::OK();
}

arrow::Status SortTableMultiColumns(const std::shared_ptr<arrow::Table> &table,
                                    const std::vector<int32_t> &sort_column_indices,
                                    arrow::MemoryPool *memory_pool,
                                    std::shared_ptr<arrow::Table> &sorted_table,
                                    const std::vector<bool> &sort_column_directions) {
  std::shared_ptr<arrow::Table> tab_to_process;  // table referenced
  // combine chunks if multiple chunks are available
  if (table->column(sort_column_indices.at(0))->num_chunks() > 1) {
    const auto &res = table->CombineChunks(memory_pool);
    RETURN_ARROW_STATUS_IF_FAILED(res.status());
    tab_to_process = res.ValueOrDie();
  } else {
    tab_to_process = table;
  }

  // sort to indices
  std::shared_ptr<arrow::UInt64Array> sorted_column_index;
  RETURN_ARROW_STATUS_IF_FAILED(cylon::SortIndicesMultiColumns(
      memory_pool, table, sort_column_indices, sorted_column_index, sort_column_directions));

  // now sort everything based on sorted index
  arrow::ArrayVector sorted_columns;
  sorted_columns.reserve(table->num_columns());

  arrow::compute::ExecContext exec_context(memory_pool);
  // no bounds check is needed as indices are guaranteed to be within range
  const arrow::compute::TakeOptions &take_options = arrow::compute::TakeOptions::NoBoundsCheck();

  for (int64_t col_index = 0; col_index < tab_to_process->num_columns(); ++col_index) {
    const arrow::Result<arrow::Datum> &res = arrow::compute::Take(
        cylon::util::GetChunkOrEmptyArray(tab_to_process->column(col_index), 0),
        sorted_column_index, take_options, &exec_context);
    RETURN_ARROW_STATUS_IF_FAILED(res.status());
    sorted_columns.emplace_back(res.ValueOrDie().make_array());
  }

  sorted_table = arrow::Table::Make(table->schema(), sorted_columns);
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

arrow::Status duplicate(const std::shared_ptr<arrow::ChunkedArray> &cArr,
                        const std::shared_ptr<arrow::Field> &field, arrow::MemoryPool *pool,
                        std::shared_ptr<arrow::ChunkedArray> &out) {
  size_t size = cArr->chunks().size();
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  for (size_t arrayIndex = 0; arrayIndex < size; arrayIndex++) {
    std::shared_ptr<arrow::Array> arr = cArr->chunk(arrayIndex);
    std::shared_ptr<arrow::ArrayData> data = arr->data();
    std::vector<std::shared_ptr<arrow::Buffer>> buffers;
    buffers.reserve(data->buffers.size());
    size_t length = cArr->length();
    for (const auto &buf : data->buffers) {
      if (buf != nullptr) {
        arrow::Result<std::shared_ptr<arrow::Buffer>> res = buf->CopySlice(0l, buf->size(), pool);
        RETURN_ARROW_STATUS_IF_FAILED(res.status());
        buffers.push_back(res.ValueOrDie());
      } else {
        buffers.push_back(nullptr);
      }
    }
    // lets send this buffer, we need to send the length at this point
    std::shared_ptr<arrow::ArrayData> new_data =
        arrow::ArrayData::Make(field->type(), length, buffers);
    std::shared_ptr<arrow::Array> array = arrow::MakeArray(data);
    arrays.push_back(array);
  }
  out = std::make_shared<arrow::ChunkedArray>(arrays, field->type());
  return arrow::Status::OK();
}

template <typename TYPE>
static inline arrow::Status sample_array(const std::shared_ptr<arrow::ChunkedArray> &ch_array,
                                         uint64_t num_samples, std::shared_ptr<arrow::Array> &out) {
  using ARROW_BUILDER_T = typename arrow::TypeTraits<TYPE>::BuilderType;
  using ARROW_ARRAY_T = typename arrow::TypeTraits<TYPE>::ArrayType;

  ARROW_BUILDER_T builder;
  auto a_status = builder.Reserve(num_samples);
  RETURN_ARROW_STATUS_IF_FAILED(a_status);

  if (num_samples > 0) {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());

    int64_t completed_samples = 0, samples_for_chunk, total_len = ch_array->length();
    for (auto &&arr : ch_array->chunks()) {
      std::shared_ptr<ARROW_ARRAY_T> casted_array = std::static_pointer_cast<ARROW_ARRAY_T>(arr);
      samples_for_chunk =
          (num_samples * casted_array->length() + total_len - 1) / total_len;  // upper bound
      samples_for_chunk = std::min(samples_for_chunk, total_len - completed_samples);

      std::uniform_int_distribution<int64_t> distrib(0, casted_array->length() - 1);
      for (int64_t i = 0; i < samples_for_chunk; i++) {
        int64_t idx = distrib(gen);
        builder.UnsafeAppend(casted_array->Value(idx));
      }
      completed_samples += samples_for_chunk;
    }

    if (builder.length() != (int64_t)num_samples) {
      return arrow::Status::ExecutionError("sampling failure");
    }
  }

  return builder.Finish(&out);
}

arrow::Status SampleTable(std::shared_ptr<arrow::Table> &table, int32_t idx, uint64_t num_samples,
                          std::shared_ptr<arrow::Array> &out) {
  return SampleArray(table->column(idx), num_samples, out);
}

arrow::Status SampleArray(const std::shared_ptr<arrow::ChunkedArray> &arr, uint64_t num_samples,
                          std::shared_ptr<arrow::Array> &out) {
  switch (arr->type()->id()) {
    case arrow::Type::BOOL:
      return sample_array<arrow::BooleanType>(arr, num_samples, out);
    case arrow::Type::UINT8:
      return sample_array<arrow::UInt8Type>(arr, num_samples, out);
    case arrow::Type::INT8:
      return sample_array<arrow::Int8Type>(arr, num_samples, out);
    case arrow::Type::UINT16:
      return sample_array<arrow::UInt16Type>(arr, num_samples, out);
    case arrow::Type::INT16:
      return sample_array<arrow::Int16Type>(arr, num_samples, out);
    case arrow::Type::UINT32:
      return sample_array<arrow::UInt32Type>(arr, num_samples, out);
    case arrow::Type::INT32:
      return sample_array<arrow::Int32Type>(arr, num_samples, out);
    case arrow::Type::UINT64:
      return sample_array<arrow::UInt32Type>(arr, num_samples, out);
    case arrow::Type::INT64:
      return sample_array<arrow::Int64Type>(arr, num_samples, out);
    case arrow::Type::FLOAT:
      return sample_array<arrow::FloatType>(arr, num_samples, out);
    case arrow::Type::DOUBLE:
      return sample_array<arrow::DoubleType>(arr, num_samples, out);
    default:
      return arrow::Status(arrow::StatusCode::Invalid, "unsupported type");
  }
}

arrow::Status SampleArray(const std::shared_ptr<arrow::Array> &arr, uint64_t num_samples,
                          std::shared_ptr<arrow::Array> &out) {
  return SampleArray(std::make_shared<arrow::ChunkedArray>(arr), num_samples, out);
}

std::shared_ptr<arrow::Array> GetChunkOrEmptyArray(const std::shared_ptr<arrow::ChunkedArray> &column, int chunk) {
  if (column->num_chunks() > 0) {
    return column->chunk(chunk);
  }
  std::shared_ptr<arrow::Array> out;
  SampleArray(column, 0, out);
  return out;
}
}  // namespace util
}  // namespace cylon

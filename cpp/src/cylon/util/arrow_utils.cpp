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
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <memory>
#include <random>
#include <vector>
#include <cmath>
#include <arrow/util/cpu_info.h>

#include <cylon/util/arrow_utils.hpp>
#include <cylon/arrow/arrow_kernels.hpp>
#include <cylon/util/macros.hpp>
#include <iostream>

namespace cylon {
namespace util {

arrow::Status SortTable(const std::shared_ptr<arrow::Table> &table, int32_t sort_column_index,
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

  for (int32_t col_index = 0; col_index < tab_to_process->num_columns(); ++col_index) {
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
  std::shared_ptr<arrow::Table> combined_tab;  // table referenced
  // combine chunks if multiple chunks are available
  if (util::CheckArrowTableContainsChunks(table, sort_column_indices)) {
    ARROW_ASSIGN_OR_RAISE(combined_tab, table->CombineChunks(memory_pool));
  } else {
    combined_tab = table;
  }

  // sort to indices
  std::shared_ptr<arrow::UInt64Array> sorted_column_index;
  RETURN_ARROW_STATUS_IF_FAILED(
      SortIndicesMultiColumns(memory_pool, combined_tab, sort_column_indices, sorted_column_index,
                              sort_column_directions));

  // now sort everything based on sorted index
  arrow::ArrayVector sorted_columns;
  sorted_columns.reserve(combined_tab->num_columns());

  arrow::compute::ExecContext exec_context(memory_pool);
  // no bounds check is needed as indices are guaranteed to be within range
  const arrow::compute::TakeOptions &take_options = arrow::compute::TakeOptions::NoBoundsCheck();

  for (int col_index = 0; col_index < combined_tab->num_columns(); ++col_index) {
    const arrow::Result<arrow::Datum> &res = arrow::compute::Take(
        cylon::util::GetChunkOrEmptyArray(combined_tab->column(col_index), 0),
        sorted_column_index, take_options, &exec_context);
    RETURN_ARROW_STATUS_IF_FAILED(res.status());
    sorted_columns.emplace_back(res.ValueOrDie().make_array());
  }

  sorted_table = arrow::Table::Make(combined_tab->schema(), sorted_columns);
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

arrow::Status Duplicate(const std::shared_ptr<arrow::ChunkedArray> &cArr, arrow::MemoryPool *pool,
                        std::shared_ptr<arrow::ChunkedArray> &out) {
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  arrays.reserve(cArr->num_chunks());
  for (const auto &arr: cArr->chunks()) {
    const std::shared_ptr<arrow::ArrayData> &data = arr->data();
    std::vector<std::shared_ptr<arrow::Buffer>> buffers;
    buffers.reserve(data->buffers.size());
    for (const auto &buf : data->buffers) {
      if (buf != nullptr) {
        const arrow::Result<std::shared_ptr<arrow::Buffer>> &res = buf->CopySlice(0l, buf->size(), pool);
        RETURN_ARROW_STATUS_IF_FAILED(res.status());
        buffers.emplace_back(res.ValueOrDie());
      } else {
        buffers.push_back(nullptr);
      }
    }
    // lets send this buffer, we need to send the length at this point
    const std::shared_ptr<arrow::ArrayData>
        &new_data = arrow::ArrayData::Make(cArr->type(), arr->length(), std::move(buffers));
    arrays.push_back(arrow::MakeArray(new_data));
  }
  out = std::make_shared<arrow::ChunkedArray>(std::move(arrays), cArr->type());
  return arrow::Status::OK();
}

arrow::Status Duplicate(const std::shared_ptr<arrow::Table> &table, arrow::MemoryPool *pool,
                        std::shared_ptr<arrow::Table> &out) {
  std::shared_ptr<arrow::Schema> schema = table->schema();
  std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;
  arrays.reserve(table->num_columns());
  for (const auto &carr: table->columns()) {
    std::shared_ptr<arrow::ChunkedArray> new_carr;
    RETURN_ARROW_STATUS_IF_FAILED(Duplicate(carr, pool, new_carr));
    arrays.push_back(std::move(new_carr));
  }

  out = arrow::Table::Make(std::move(schema), std::move(arrays));

  return arrow::Status::OK();
}

template<typename TYPE>
static inline arrow::Status sample_fixed_size_array(const std::shared_ptr<arrow::ChunkedArray> &ch_array,
                                                    uint64_t num_samples,
                                                    std::shared_ptr<arrow::Array> &out,
                                                    arrow::MemoryPool *pool) {
  using ARROW_BUILDER_T = typename arrow::TypeTraits<TYPE>::BuilderType;
  using ARROW_ARRAY_T = typename arrow::TypeTraits<TYPE>::ArrayType;

  ARROW_BUILDER_T builder(ch_array->type(), pool);
  RETURN_ARROW_STATUS_IF_FAILED(builder.Reserve(num_samples));

  if (num_samples == 0) {
    return builder.Finish(&out);
  }

  // if the num_samples is greater than the array length, dont sample, just return the array
  if ((int64_t) num_samples >= ch_array->length()) {
    if (ch_array->num_chunks() > 1) {
      const auto &res = arrow::Concatenate(ch_array->chunks(), pool);
      RETURN_ARROW_STATUS_IF_FAILED(res.status());
      out = res.ValueOrDie();
    } else {
      out = ch_array->chunk(0);
    }
    return arrow::Status::OK();
  }

  // general case - using our own take method here because sampling in chunked arrays is not so efficient in arrow
  static std::random_device rd;
  static std::mt19937_64 gen(rd());

  int64_t completed_samples = 0, samples_for_chunk, total_len = ch_array->length();
  for (auto &&arr : ch_array->chunks()) {
    std::shared_ptr<ARROW_ARRAY_T> casted_array = std::static_pointer_cast<ARROW_ARRAY_T>(arr);
    samples_for_chunk = (num_samples * casted_array->length() + total_len - 1) / total_len;  // upper bound
    samples_for_chunk = std::min(samples_for_chunk, total_len - completed_samples);

    std::uniform_int_distribution<int64_t> distrib(0, casted_array->length() - 1);
    for (int64_t i = 0; i < samples_for_chunk; i++) {
      builder.UnsafeAppend(casted_array->Value(distrib(gen)));
    }
    completed_samples += samples_for_chunk;
  }

  if (builder.length() != (int64_t) num_samples) {
    return arrow::Status::ExecutionError("sampling failure");
  }

  return builder.Finish(&out);
}

template<typename TYPE>
static inline arrow::Status sample_binary_array(const std::shared_ptr<arrow::ChunkedArray> &ch_array,
                                                uint64_t num_samples,
                                                std::shared_ptr<arrow::Array> &out,
                                                arrow::MemoryPool *pool) {
  using ARROW_BUILDER_T = typename arrow::TypeTraits<TYPE>::BuilderType;

  // if num_samples == 0, just finish builder
  if (num_samples == 0) {
    ARROW_BUILDER_T builder(ch_array->type(), pool);
    RETURN_ARROW_STATUS_IF_FAILED(builder.Reserve(0));
    return builder.Finish(&out);
  }

  // if the num_samples is greater than the array length, dont sample, just return the array
  if ((int64_t) num_samples >= ch_array->length()) {
    if (ch_array->num_chunks() > 1) {
      const auto &res = arrow::Concatenate(ch_array->chunks(), pool);
      RETURN_ARROW_STATUS_IF_FAILED(res.status());
      out = res.ValueOrDie();
    } else {
      out = ch_array->chunk(0);
    }
    return arrow::Status::OK();
  }

  // general case
  static std::random_device rd;
  static std::mt19937_64 gen(rd());
  std::uniform_int_distribution<int64_t> distrib(0, ch_array->length() - 1);

  arrow::Int64Builder builder(pool);
  RETURN_ARROW_STATUS_IF_FAILED(builder.Reserve((int64_t) num_samples));
  for (uint64_t i = 0; i < num_samples; i++) {
    builder.UnsafeAppend(distrib(gen));
  }

  const arrow::Result<arrow::Datum>
      &res = arrow::compute::Take(ch_array, builder.Finish().ValueOrDie(), arrow::compute::TakeOptions(false));
  RETURN_ARROW_STATUS_IF_FAILED(res.status());

  out = res.ValueOrDie().make_array();

  return arrow::Status::OK();
}

arrow::Status SampleArray(std::shared_ptr<arrow::Table> &table,
                          int32_t idx,
                          uint64_t num_samples,
                          std::shared_ptr<arrow::Array> &out,
                          arrow::MemoryPool *pool) {
  return SampleArray(table->column(idx), num_samples, out, pool);
}

arrow::Status SampleArray(const std::shared_ptr<arrow::ChunkedArray> &arr,
                          uint64_t num_samples,
                          std::shared_ptr<arrow::Array> &out,
                          arrow::MemoryPool *pool) {
  switch (arr->type()->id()) {
    case arrow::Type::BOOL:return sample_fixed_size_array<arrow::BooleanType>(arr, num_samples, out, pool);
    case arrow::Type::UINT8:return sample_fixed_size_array<arrow::UInt8Type>(arr, num_samples, out, pool);
    case arrow::Type::INT8:return sample_fixed_size_array<arrow::Int8Type>(arr, num_samples, out, pool);
    case arrow::Type::UINT16:return sample_fixed_size_array<arrow::UInt16Type>(arr, num_samples, out, pool);
    case arrow::Type::INT16:return sample_fixed_size_array<arrow::Int16Type>(arr, num_samples, out, pool);
    case arrow::Type::UINT32:return sample_fixed_size_array<arrow::UInt32Type>(arr, num_samples, out, pool);
    case arrow::Type::INT32:return sample_fixed_size_array<arrow::Int32Type>(arr, num_samples, out, pool);
    case arrow::Type::UINT64:return sample_fixed_size_array<arrow::UInt32Type>(arr, num_samples, out, pool);
    case arrow::Type::INT64:return sample_fixed_size_array<arrow::Int64Type>(arr, num_samples, out, pool);
    case arrow::Type::FLOAT:return sample_fixed_size_array<arrow::FloatType>(arr, num_samples, out, pool);
    case arrow::Type::DOUBLE:return sample_fixed_size_array<arrow::DoubleType>(arr, num_samples, out, pool);
    case arrow::Type::DATE32:return sample_fixed_size_array<arrow::Date32Type>(arr, num_samples, out, pool);
    case arrow::Type::DATE64:return sample_fixed_size_array<arrow::Date64Type>(arr, num_samples, out, pool);
    case arrow::Type::TIMESTAMP:return sample_fixed_size_array<arrow::TimestampType>(arr, num_samples, out, pool);
    case arrow::Type::TIME32:return sample_fixed_size_array<arrow::Time32Type>(arr, num_samples, out, pool);
    case arrow::Type::TIME64:return sample_fixed_size_array<arrow::Time64Type>(arr, num_samples, out, pool);
    case arrow::Type::STRING: return sample_binary_array<arrow::StringType>(arr, num_samples, out, pool);
    case arrow::Type::BINARY:return sample_binary_array<arrow::BinaryType>(arr, num_samples, out, pool);
    case arrow::Type::FIXED_SIZE_BINARY:
      return sample_fixed_size_array<arrow::FixedSizeBinaryType>(arr,
                                                                 num_samples,
                                                                 out,
                                                                 pool);
    default:return arrow::Status(arrow::StatusCode::Invalid, "unsupported type");
  }
}

arrow::Status SampleArray(const std::shared_ptr<arrow::Array> &arr,
                          uint64_t num_samples,
                          std::shared_ptr<arrow::Array> &out,
                          arrow::MemoryPool *pool) {
  return SampleArray(std::make_shared<arrow::ChunkedArray>(arr), num_samples, out, pool);
}

std::shared_ptr<arrow::Array> GetChunkOrEmptyArray(const std::shared_ptr<arrow::ChunkedArray> &column, int chunk,
                                                   arrow::MemoryPool *pool) {
  if (column->num_chunks() > 0) {
    return column->chunk(chunk);
  }
  auto res = arrow::MakeArrayOfNull(column->type(), 0, pool);
  return res.ok() ? res.ValueOrDie() : nullptr;
}

uint64_t GetNumberSplitsToFitInCache(int64_t total_bytes, int total_elements, int parallel) {
  if (total_elements == 0 || total_bytes == 0) {
    return 1;
  }

  int64_t cache_size = arrow::internal::CpuInfo::GetInstance()->CacheSize(arrow::internal::CpuInfo::L1_CACHE);
  cache_size += arrow::internal::CpuInfo::GetInstance()->CacheSize(arrow::internal::CpuInfo::L2_CACHE);
  int64_t average_element_size = total_bytes / total_elements;
  int64_t elements_in_cache = cache_size / average_element_size;
  return ceil((double)(total_elements / parallel) / elements_in_cache);
}

std::array<int64_t, 2> GetBytesAndElements(std::shared_ptr<arrow::Table> table, const std::vector<int> &columns) {
  int64_t num_elements = 0;
  int64_t num_bytes = 0;
  for (int64_t t : columns) {
    const std::shared_ptr<arrow::ChunkedArray> &ptr = table->column(t);
    for (std::shared_ptr<arrow::Array> arr : ptr->chunks()) {
      num_elements += arr->length();
      for (auto &b: arr->data()->buffers) {
        if (b != nullptr) {
          num_bytes += b->size();
        }
      }
    }
  }
  return {num_elements, num_bytes};
}
arrow::Status CreateEmptyTable(const std::shared_ptr<arrow::Schema> &schema,
                               std::shared_ptr<arrow::Table> *output,
                               arrow::MemoryPool *pool) {
  std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;
  arrays.reserve(schema->num_fields());

  for (int i = 0; i < schema->num_fields(); i++){
    const auto& t = schema->field(i)->type();
    ARROW_ASSIGN_OR_RAISE(auto arr, arrow::MakeArrayOfNull(t, 0, pool))
    arrays.emplace_back(std::make_shared<arrow::ChunkedArray>(std::move(arr)));
  }

  *output = arrow::Table::Make(schema, std::move(arrays), 0);
  return arrow::Status::OK();
}

arrow::Status MakeEmptyArrowTable(const std::shared_ptr<arrow::Schema> &schema,
                                  std::shared_ptr<arrow::Table> *table,
                                  arrow::MemoryPool *pool) {
  arrow::ChunkedArrayVector arrays;
  arrays.reserve(schema->num_fields());
  for (const auto &f: schema->fields()) {
    ARROW_ASSIGN_OR_RAISE(auto arr, arrow::MakeArrayOfNull(f->type(), 0, pool))
    arrays.push_back(std::make_shared<arrow::ChunkedArray>(std::move(arr)));
  }
  *table = arrow::Table::Make(schema, std::move(arrays), 0);
  return arrow::Status::OK();
}

bool CheckArrowTableContainsChunks(const std::shared_ptr<arrow::Table> &table,
                                   const std::vector<int> &columns) {
  if (columns.empty()) {
    return std::any_of(table->columns().begin(), table->columns().end(),
                       [](const auto &col) { return col->num_chunks() > 1; });
  } else {
    return std::any_of(columns.begin(), columns.end(),
                       [&](int i) { return table->column(i)->num_chunks() > 1; });
  }
}

arrow::Status MakeDummyArray(const std::shared_ptr<arrow::DataType> &type, int64_t num_elems,
                             std::shared_ptr<arrow::Array> *out, arrow::MemoryPool *pool) {
  std::unique_ptr<arrow::ArrayBuilder> builder;
  RETURN_NOT_OK(arrow::MakeBuilder(pool, type, &builder));
  RETURN_NOT_OK(builder->AppendEmptyValues(num_elems));
  return builder->Finish(out);
}

}  // namespace util
}  // namespace cylon

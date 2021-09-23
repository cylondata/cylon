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
#include <arrow/compute/api_vector.h>

#include "cylon/indexing/indexer.hpp"
#include "cylon/util/macros.hpp"
#include "cylon/ctx/arrow_memory_pool_utils.hpp"
#include "cylon/util/arrow_utils.hpp"
#include "cylon/indexing/index_utils.hpp"

namespace cylon {
namespace indexing {

Status Loc(const std::shared_ptr<Table> &input_table,
           const std::shared_ptr<arrow::Scalar> &start_value,
           const std::shared_ptr<arrow::Scalar> &end_value,
           const std::vector<int> &columns,
           std::shared_ptr<cylon::Table> *output) {
  int64_t start = 0, end = 0;
  const auto &index = input_table->GetArrowIndex();
  RETURN_CYLON_STATUS_IF_FAILED(index->LocationRangeByValue(start_value, end_value, &start, &end));
  RETURN_CYLON_STATUS_IF_FAILED(SliceTableByRange(input_table, start, end, columns, output, false));
  return Status::OK();
}

Status Loc(const std::shared_ptr<Table> &input_table,
           const std::shared_ptr<arrow::Array> &values,
           const std::vector<int> &columns,
           std::shared_ptr<cylon::Table> *output) {
  const auto &index = input_table->GetArrowIndex();

  std::shared_ptr<arrow::Int64Array> indices;
  RETURN_CYLON_STATUS_IF_FAILED(index->LocationByVector(values, &indices));

  return SelectTableByRows(input_table, std::move(indices), columns, output, false);
}

Status iLoc(const std::shared_ptr<Table> &input_table,
            const std::shared_ptr<arrow::Scalar> &start_index,
            const std::shared_ptr<arrow::Scalar> &end_index,
            const std::vector<int> &columns,
            std::shared_ptr<Table> *output) {
  std::shared_ptr<arrow::Scalar> casted_start_index, casted_end_index;
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(start_index->CastTo(arrow::int64()).Value(&casted_start_index));
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(end_index->CastTo(arrow::int64()).Value(&casted_end_index));

  int64_t start = util::ArrowScalarValue<arrow::Int64Type>::Extract(casted_start_index);
  int64_t end = util::ArrowScalarValue<arrow::Int64Type>::Extract(casted_end_index);

  return SliceTableByRange(input_table, start, end, columns, output, false);
}

Status iLoc(const std::shared_ptr<Table> &input_table,
            const std::shared_ptr<arrow::Array> &indices,
            const std::vector<int> &columns,
            std::shared_ptr<cylon::Table> *output) {
  std::shared_ptr<arrow::Array> casted_indices;
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(arrow::compute::Cast(*indices, arrow::int64()).Value(&casted_indices));

  return SelectTableByRows(input_table, std::move(casted_indices), columns, output, false);
}

Status Loc(const std::shared_ptr<Table> &input_table,
           const std::shared_ptr<arrow::Scalar> &start_index,
           const std::shared_ptr<arrow::Scalar> &end_index,
           int column_index,
           std::shared_ptr<Table> *output) {
  return Loc(input_table, start_index, end_index, std::vector<int>{column_index}, output);
}

Status Loc(const std::shared_ptr<Table> &input_table,
           const std::shared_ptr<arrow::Scalar> &start_index,
           const std::shared_ptr<arrow::Scalar> &end_index,
           int start_column,
           int end_column,
           std::shared_ptr<Table> *output) {
  if (end_column < start_column) {
    return {Code::Invalid, "end column < start column"};
  }
  std::vector<int> columns(end_column - start_column + 1); // +1 for inclusive range
  std::iota(columns.begin(), columns.end(), start_column);

  return Loc(input_table, start_index, end_index, columns, output);
}

Status Loc(const std::shared_ptr<Table> &input_table,
           const std::shared_ptr<arrow::Array> &indices,
           int column_index,
           std::shared_ptr<cylon::Table> *output) {
  return Loc(input_table, indices, std::vector<int>{column_index}, output);
}
Status Loc(const std::shared_ptr<Table> &input_table,
           const std::shared_ptr<arrow::Array> &indices,
           int start_column,
           int end_column,
           std::shared_ptr<cylon::Table> *output) {
  if (end_column < start_column) {
    return {Code::Invalid, "end column < start column"};
  }
  std::vector<int> columns(end_column - start_column + 1); // +1 for inclusive range
  std::iota(columns.begin(), columns.end(), start_column);

  return Loc(input_table, indices, columns, output);
}

Status iLoc(const std::shared_ptr<Table> &input_table,
            const std::shared_ptr<arrow::Scalar> &start_index,
            const std::shared_ptr<arrow::Scalar> &end_index,
            int column_index,
            std::shared_ptr<Table> *output) {
  return iLoc(input_table, start_index, end_index, std::vector<int>{column_index}, output);
}

Status iLoc(const std::shared_ptr<Table> &input_table,
            const std::shared_ptr<arrow::Scalar> &start_index,
            const std::shared_ptr<arrow::Scalar> &end_index,
            int start_column,
            int end_column,
            std::shared_ptr<cylon::Table> *output) {
  if (end_column < start_column) {
    return {Code::Invalid, "end column < start column"};
  }
  std::vector<int> columns(end_column - start_column + 1); // +1 for inclusive range
  std::iota(columns.begin(), columns.end(), start_column);

  return iLoc(input_table, start_index, end_index, columns, output);
}

Status iLoc(const std::shared_ptr<Table> &input_table,
            const std::shared_ptr<arrow::Array> &indices,
            int column_index,
            std::shared_ptr<cylon::Table> *output) {
  return iLoc(input_table, indices, std::vector<int>{column_index}, output);
}

Status iLoc(const std::shared_ptr<Table> &input_table,
            const std::shared_ptr<arrow::Array> &indices,
            int start_column,
            int end_column,
            std::shared_ptr<cylon::Table> *output) {
  if (end_column < start_column) {
    return {Code::Invalid, "end column < start column"};
  }
  std::vector<int> columns(end_column - start_column + 1); // +1 for inclusive range
  std::iota(columns.begin(), columns.end(), start_column);

  return iLoc(input_table, indices, columns, output);
}

} // namespace indexing
} // namespace cylon
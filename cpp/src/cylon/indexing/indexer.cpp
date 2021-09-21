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

#include "cylon/indexing/indexer.hpp"
#include "cylon/util/macros.hpp"
#include "cylon/ctx/arrow_memory_pool_utils.hpp"
#include "cylon/util/arrow_utils.hpp"

namespace cylon {
namespace indexing {

//bool CheckIsIndexValueUnique(const std::shared_ptr<arrow::Scalar> &index_value,
//                             const std::shared_ptr<BaseArrowIndex> &index) {
//  if (index->GetIndexingType() == cylon::IndexingType::Range) {
//    return true;
//  } else {
//    const auto &index_arr = index->GetIndexArray();
//    int64_t find_cout = 0;
//    for (int64_t ix = 0; ix < index_arr->length(); ix++) {
//      auto val = index_arr->GetScalar(ix).ValueOrDie();
//      auto index_value_cast = index_value->CastTo(val->type).ValueOrDie();
//      if (val == index_value_cast) {
//        find_cout++;
//      }
//      if (find_cout > 1) {
//        return false;
//      }
//    }
//    return true;
//  }
//
//}

// we need to ensure that the index col id is included in the columns vector
// if not add it to columns vector
int update_vector_with_index_column(int curr_index_col, std::vector<int> *columns) {
  // found_idx will be the position of the curr_index_col.
  size_t found_idx = std::distance(columns->begin(), std::find(columns->begin(), columns->end(), curr_index_col));
  if (found_idx >= columns->size()) {
    // curr_index_col not found! push it into columns vector
    columns->push_back(curr_index_col);
    return static_cast<int>(columns->size() - 1);
  } else {
    return static_cast<int>(found_idx);
  }
}

Status SliceTableByRange(const std::shared_ptr<Table> &input_table,
                         int64_t start,
                         int64_t end_inclusive,
                         std::vector<int> columns,
                         std::shared_ptr<Table> *output,
                         bool reset_index) {
  const auto &arrow_t = input_table->get_table();
  const auto &ctx = input_table->GetContext();

  const auto &curr_index = input_table->GetArrowIndex();

  int found_idx = -1;
  if (!reset_index && curr_index->GetIndexingType() != Range) {
    found_idx = update_vector_with_index_column(input_table->GetArrowIndex()->col_id(), &columns);
  }

  // select and slice the table
  const auto &res = arrow_t->SelectColumns(columns);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(res.status());

  const auto &selected_table = res.ValueOrDie();

  // then slice. + 1 added include the end boundary
  auto out_arrow_table = selected_table->Slice(start, (end_inclusive - start + 1));

  RETURN_CYLON_STATUS_IF_FAILED(Table::FromArrowTable(ctx, std::move(out_arrow_table), *output));

  // now fix the index
  if (reset_index) { // if reset index, nothing to do
    return Status::OK();
  }

  if (curr_index->GetIndexingType() == Range) {
    std::shared_ptr<BaseArrowIndex> sliced_index;
    // trivially create a new range index (+ 1 added include the end boundary)
    RETURN_CYLON_STATUS_IF_FAILED(
        std::static_pointer_cast<ArrowRangeIndex>(curr_index)->Slice(start, end_inclusive, &sliced_index));

    return (*output)->SetArrowIndex(std::move(sliced_index));
  } else {
    // build index from found_idx column
    return (*output)->SetArrowIndex(static_cast<int>(found_idx), curr_index->GetIndexingType());
  }
}

/*
 * Filter table based on indices. The new table will inherit the indices array as it's index (LinearIndex)
 */
Status FilterTable(const std::shared_ptr<Table> &input_table,
                   const std::shared_ptr<arrow::Array> &indices,
                   std::vector<int> columns,
                   std::shared_ptr<Table> *output,
                   bool bounds_check,
                   bool reset_index) {
  const auto &arrow_t = input_table->get_table();
  const auto &ctx = input_table->GetContext();

  const auto &curr_index = input_table->GetArrowIndex();

  int found_idx = -1;
  if (!reset_index && curr_index->GetIndexingType() != Range) {
    found_idx = update_vector_with_index_column(input_table->GetArrowIndex()->col_id(), &columns);
  }

  //  first filter columns
  const auto &res = arrow_t->SelectColumns(columns);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(res.status());

  // then call take on the result
  const auto &take_res = arrow::compute::Take(res.ValueOrDie(), indices,
                                              bounds_check ? arrow::compute::TakeOptions::BoundsCheck()
                                                           : arrow::compute::TakeOptions::NoBoundsCheck());
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(take_res.status());
  RETURN_CYLON_STATUS_IF_FAILED(Table::FromArrowTable(ctx, take_res.ValueOrDie().table(), *output));

  // now fix the index
  if (reset_index) { // if reset index, nothing to do
    return Status::OK();
  }

  const auto &index = input_table->GetArrowIndex();
  std::shared_ptr<BaseArrowIndex> new_index;
  if (index->GetIndexingType() == Range) {
    // if the indexing type is range, we can use the indices array as a linear index to the table
    return (*output)->SetArrowIndex(indices, Linear);
  } else {
    return (*output)->SetArrowIndex(static_cast<int>(found_idx), curr_index->GetIndexingType());
  }
}

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

  return FilterTable(input_table, std::move(indices), columns, output, false);
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

  return FilterTable(input_table, std::move(casted_indices), columns, output, false);
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
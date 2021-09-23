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

#include <arrow/util/bitmap_ops.h>

#include "cylon/util/macros.hpp"
#include "cylon/ctx/arrow_memory_pool_utils.hpp"
#include "index_utils.hpp"

namespace cylon {
namespace indexing {

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
Status SelectTableByRows(const std::shared_ptr<Table> &input_table,
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

Status FilterTableByMask(const std::shared_ptr<Table> &input_table,
                         const std::shared_ptr<arrow::Array> &mask,
                         std::vector<int> columns,
                         std::shared_ptr<Table> *output,
                         bool reset_index) {
  if (mask->type_id() != arrow::Type::BOOL) {
    return {Code::Invalid, "mask should be Boolean type"};
  }

  if (input_table->Rows() != mask->length()) {
    return {Code::Invalid, "mask should match the table length"};
  }

  const auto &res = arrow::compute::internal::GetTakeIndices(*mask->data(),
                                                             arrow::compute::FilterOptions::NullSelectionBehavior::DROP,
                                                             ToArrowPool(input_table->GetContext()));
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(res.status());

  return SelectTableByRows(input_table,
                           arrow::MakeArray(res.ValueOrDie()),
                           std::move(columns),
                           output,
      /*bounds_check=*/false,
                           reset_index);
}

Status MaskTable(const std::shared_ptr<Table> &input_table,
                 const std::shared_ptr<Table> &mask,
                 std::shared_ptr<Table> *output) {
  if (input_table->Columns() != mask->Columns() || input_table->Rows() != mask->Rows()) {
    return {Code::Invalid, "input table and mask dimensions don't match"};
  }

  for (auto &&c: mask->get_table()->columns()) {
    if (c->type()->id() != arrow::Type::BOOL) {
      return {Code::Invalid, "mask should be all boolean columns"};
    }
  }

  if (input_table->Empty()) {
    *output = input_table;
    return Status::OK();
  }

  arrow::ArrayVector output_arrays;
  output_arrays.reserve(input_table->Columns());
  arrow::MemoryPool *pool = ToArrowPool(input_table->GetContext());

  const auto &res = input_table->get_table()->CombineChunks(pool);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(res.status());
  auto arrow_table = res.ValueOrDie();

  const auto &mask_res = mask->get_table()->CombineChunks(pool);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(mask_res.status());
  const auto &arrow_mask = mask_res.ValueOrDie();

  for (int i = 0; i < arrow_table->num_columns(); i++) {
    const auto &curr_data = arrow_table->column(i)->chunk(0)->data();
    const auto &mask_data = arrow_mask->column(i)->chunk(0)->data();

    std::vector<std::shared_ptr<arrow::Buffer>> new_buffers;
    new_buffers.reserve(curr_data->buffers.size());

    // fix the validity buffer
    if (curr_data->MayHaveNulls()) {
      // create a new bitmap with curr_data's offset
      const auto &res_and = arrow::internal::BitmapAnd(pool,
                                                       curr_data->buffers[0]->data(),
                                                       curr_data->offset,
                                                       mask_data->buffers[1]->data(),
                                                       mask_data->offset,
                                                       curr_data->length,
                                                       curr_data->offset);
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(res_and.status());
      new_buffers.emplace_back(res_and.ValueOrDie());
    } else {
      // create a new bitmap with curr_data's offset
      const auto &res_buf = arrow::AllocateBitmap(curr_data->length + curr_data->offset, pool);
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(res_buf.status());
      auto new_buff = res_buf.ValueOrDie();

      arrow::internal::CopyBitmap(mask_data->buffers[1]->data(), mask_data->offset, mask_data->length,
                                  new_buff->mutable_data(), curr_data->offset);
      new_buffers.emplace_back(std::move(new_buff));
    }

    // simply copy the buffers to new buffers
    for (size_t b = 1; b < curr_data->buffers.size(); b++) {
      new_buffers.push_back(curr_data->buffers[b]);
    }

    auto new_data = arrow::ArrayData::Make(curr_data->type,
                                           curr_data->length,
                                           std::move(new_buffers),
                                           arrow::kUnknownNullCount,
                                           curr_data->offset);
    output_arrays.push_back(arrow::MakeArray(new_data));
  }

  auto new_table = arrow::Table::Make(arrow_table->schema(), output_arrays);

  RETURN_CYLON_STATUS_IF_FAILED(Table::FromArrowTable(input_table->GetContext(),
                                                      std::move(new_table), *output));

  const auto &curr_index = input_table->GetArrowIndex();
  return (*output)->SetArrowIndex(curr_index->col_id(), curr_index->GetIndexingType());
}

}
}
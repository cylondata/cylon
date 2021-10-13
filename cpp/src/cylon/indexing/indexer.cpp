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
#include <cylon/indexing/indexer.hpp>
#include <cylon/indexing/index_utils.hpp>

cylon::Status BuildArrowIndexFromArrayByKernel(cylon::IndexingType indexing_type,
											   std::shared_ptr<arrow::Array> &sub_index_arr,
											   arrow::MemoryPool *pool,
											   std::shared_ptr<cylon::BaseArrowIndex> &loc_index) {

  switch (indexing_type) {
    case cylon::Range:
      return cylon::IndexUtil::BuildArrowRangeIndexFromArray(sub_index_arr->length(), pool, loc_index);
    case cylon::Linear:
      return cylon::IndexUtil::BuildArrowLinearIndexFromArrowArray(sub_index_arr, pool, loc_index);
    case cylon::Hash:
      return cylon::IndexUtil::BuildArrowHashIndexFromArray(sub_index_arr, pool, loc_index);
    case cylon::BinaryTree:
      return cylon::Status(cylon::Code::NotImplemented, "Binary Tree Indexing not implemented!");
    case cylon::BTree:
      return cylon::Status(cylon::Code::NotImplemented, "B-Tree Indexing not implemented!");
  }
  return cylon::Status(cylon::Code::TypeError, "Unknown indexing type.");
}

cylon::Status SetArrowIndexForLocResultTable(const std::shared_ptr<cylon::BaseArrowIndex> &index,
											 std::vector<int64_t> &sub_index_locations,
											 std::shared_ptr<cylon::Table> &output,
											 cylon::IndexingType indexing_type) {

  std::shared_ptr<cylon::BaseArrowIndex> loc_index;
  std::shared_ptr<arrow::Array> sub_index_pos_arr;
  std::shared_ptr<arrow::Array> sub_index_arr;
  const auto& ctx = output->GetContext();
  auto const pool = cylon::ToArrowPool(ctx);
  auto const index_arr = index->GetIndexArray();
  arrow::Int64Builder builder(pool);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.AppendValues(sub_index_locations));
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.Finish(&sub_index_pos_arr));
  arrow::Result<arrow::Datum> datum = arrow::compute::Take(index_arr, sub_index_pos_arr);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(datum.status());
  sub_index_arr = datum.ValueOrDie().make_array();
  RETURN_CYLON_STATUS_IF_FAILED(BuildArrowIndexFromArrayByKernel(indexing_type, sub_index_arr, pool, loc_index));
  RETURN_CYLON_STATUS_IF_FAILED(output->SetArrowIndex(loc_index, false));
  return cylon::Status::OK();
}

cylon::Status SetArrowIndexForLocResultTable(const std::shared_ptr<cylon::BaseArrowIndex> &index,
											 int64_t start_pos,
											 int64_t end_pos,
											 std::shared_ptr<cylon::Table> &output,
											 cylon::IndexingType indexing_type) {
  std::shared_ptr<cylon::BaseArrowIndex> loc_index;
  std::shared_ptr<arrow::Array> sub_index_arr;
  const auto& ctx = output->GetContext();
  auto const pool = cylon::ToArrowPool(ctx);
  auto const index_arr = index->GetIndexArray();
  sub_index_arr = index_arr->Slice(start_pos, (end_pos - start_pos + 1));
  RETURN_CYLON_STATUS_IF_FAILED(BuildArrowIndexFromArrayByKernel(indexing_type, sub_index_arr, pool, loc_index));
  RETURN_CYLON_STATUS_IF_FAILED(output->SetArrowIndex(loc_index, false));
  return cylon::Status::OK();

}

cylon::Status GetArrowLocFilterIndices(const std::shared_ptr<arrow::Scalar> &start_index,
									   const std::shared_ptr<arrow::Scalar> &end_index,
									   const std::shared_ptr<cylon::BaseArrowIndex> &index,
									   int64_t *s_index,
									   int64_t *e_index) {
  std::shared_ptr<arrow::Table> out_artb;
  if (!cylon::CheckIsIndexValueUnique(start_index, index) || !cylon::CheckIsIndexValueUnique(end_index, index)) {
	return cylon::Status(cylon::Code::IndexError, "Start index must be unique");
  }
  RETURN_CYLON_STATUS_IF_FAILED(index->LocationByValue(start_index, s_index));
  RETURN_CYLON_STATUS_IF_FAILED(index->LocationByValue(end_index, e_index));
  return cylon::Status::OK();
}

cylon::Status SliceTableByRange(int64_t start_index,
								int64_t end_index,
								const std::shared_ptr<cylon::Table> &input_table,
								std::shared_ptr<cylon::Table> &output) {

  std::shared_ptr<arrow::Table> out_artb;
  auto artb = input_table->get_table();
  const auto& ctx = input_table->GetContext();
  // + 1 added include the end boundary
  out_artb = artb->Slice(start_index, (end_index - start_index + 1));
  RETURN_CYLON_STATUS_IF_FAILED(cylon::Table::FromArrowTable(ctx, out_artb, output));
  return cylon::Status::OK();
}

cylon::Status GetColumnIndicesFromLimits(const int &start_column,
										 const int &end_column,
										 std::vector<int> &selected_columns) {

  if (start_column > end_column) {
    return cylon::Status(cylon::Code::Invalid, "Invalid column boundaries");
  }
  for (int s = start_column; s <= end_column; s++) {
    selected_columns.push_back(s);
  }
  return cylon::Status::OK();
}

cylon::Status FilterColumnsFromTable(const std::shared_ptr<cylon::Table> &input_table,
									 const std::vector<int> &filter_columns,
									 std::shared_ptr<cylon::Table> &output) {
  const auto& ctx = input_table->GetContext();
  arrow::Result<std::shared_ptr<arrow::Table>> result = input_table->get_table()->SelectColumns(filter_columns);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
  RETURN_CYLON_STATUS_IF_FAILED(cylon::Table::FromArrowTable(ctx, result.ValueOrDie(), output));
  return cylon::Status::OK();
}

static cylon::Status ResolveArrowILocIndices(const std::shared_ptr<arrow::Array> &input_indices,
											 std::vector<int64_t> &output_indices) {
  std::shared_ptr<arrow::Int64Array> input_indices_ar = std::static_pointer_cast<arrow::Int64Array>(input_indices);
  for (int64_t ix = 0; ix < input_indices->length(); ix++) {
    int64_t val = input_indices_ar->Value(ix);
    output_indices[ix] = val;
  }
  return cylon::Status::OK();
}

cylon::Status ResolveArrowLocIndices(const std::shared_ptr<arrow::Array> &input_indices,
									 const std::shared_ptr<cylon::BaseArrowIndex> &index,
									 std::vector<int64_t> &output_indices) {
  RETURN_CYLON_STATUS_IF_FAILED(index->LocationByVector(input_indices, output_indices));
  return cylon::Status::OK();
}

cylon::Status GetTableFromIndices(const std::shared_ptr<cylon::Table> &input_table,
								  const std::vector<int64_t> &filter_indices,
								  std::shared_ptr<cylon::Table> &output) {

  std::shared_ptr<arrow::Array> out_idx;
  const auto& ctx = input_table->GetContext();
  auto const pool = cylon::ToArrowPool(ctx);
  arrow::compute::ExecContext fn_ctx(pool);
  arrow::Int64Builder idx_builder(pool);
  auto const arrow_table = input_table->get_table();
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(idx_builder.AppendValues(filter_indices));
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(idx_builder.Finish(&out_idx));
  arrow::Result<arrow::Datum> result = arrow::compute::Take(arrow_table,
															out_idx,
															arrow::compute::TakeOptions::Defaults(),
															&fn_ctx);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
  std::shared_ptr<arrow::Table> filter_table;
  filter_table = result.ValueOrDie().table();
  RETURN_CYLON_STATUS_IF_FAILED(cylon::Table::FromArrowTable(ctx, filter_table, output));
  return cylon::Status::OK();
}

cylon::Status GetTableFromArrayIndices(const std::shared_ptr<cylon::Table> &input_table,
									   const std::shared_ptr<arrow::Int64Array> &filter_indices,
									   std::shared_ptr<cylon::Table> &output) {

  std::shared_ptr<arrow::Array> out_idx;
  const auto& ctx = input_table->GetContext();
  auto const pool = cylon::ToArrowPool(ctx);
  auto const arrow_table = input_table->get_table();
  arrow::compute::ExecContext fn_ctx(pool);
  arrow::Result<arrow::Datum> result = arrow::compute::Take(arrow_table,
															filter_indices,
															arrow::compute::TakeOptions::Defaults(),
															&fn_ctx);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
  std::shared_ptr<arrow::Table> filter_table;
  filter_table = result.ValueOrDie().table();
  RETURN_CYLON_STATUS_IF_FAILED(cylon::Table::FromArrowTable(ctx, filter_table, output));
  return cylon::Status::OK();
}

/**
 * ILocIndexer implementations
 * */

/**
   * Implementation Note for feature enhancement
   *
   * When multi-indexing is supported. We can keep a range-index and any user provided multi-index schemas
   * In such a scenario ILoc operations can be faciliated by calling the super class (LocIndexer) as ILocIndexer
   * is extended from it. Here multi-indexing handling must be done
   *
   *    Example
   *    -------
   *    LOG(INFO) << "Calling ILoc operation mode 1";
   *    status = cylon::LocIndexer::loc(start_index, end_index, start_column_index, end_column_index, input_table, output);
   *
   *    if (!status.is_ok()) {
   *      LOG(ERROR) << "Error occurred in iloc operation";
   *      return status;
   *    }
   *
   * */


bool cylon::CheckIsIndexValueUnique(const std::shared_ptr<arrow::Scalar> &index_value,
									const std::shared_ptr<BaseArrowIndex> &index) {

  if (index->GetIndexingType() == cylon::IndexingType::Range) {
    return true;
  } else {
    auto index_arr = index->GetIndexArray();
    int64_t find_cout = 0;
    for (int64_t ix = 0; ix < index_arr->length(); ix++) {
      auto val = index_arr->GetScalar(ix).ValueOrDie();
      auto index_value_cast = index_value->CastTo(val->type).ValueOrDie();
      if (val == index_value_cast) {
        find_cout++;
      }
      if (find_cout > 1) {
        return false;
      }
    }
    return true;
  }

}

/*
 * Arrow Input based Loc operators
 *
 * **/

cylon::Status cylon::ArrowLocIndexer::loc(const std::shared_ptr<arrow::Scalar> &start_index,
										  const std::shared_ptr<arrow::Scalar> &end_index,
										  const int column_index,
										  const std::shared_ptr<Table> &input_table,
										  std::shared_ptr<cylon::Table> &output) {
  auto index = input_table->GetArrowIndex();
  std::shared_ptr<cylon::Table> temp_output;
  int64_t s_index = -1;
  int64_t e_index = -1;
  RETURN_CYLON_STATUS_IF_FAILED(GetArrowLocFilterIndices(start_index, end_index, index, &s_index, &e_index));
  std::vector<int> filter_columns = {column_index};
  RETURN_CYLON_STATUS_IF_FAILED(SliceTableByRange(s_index, e_index, input_table, temp_output));
  RETURN_CYLON_STATUS_IF_FAILED(FilterColumnsFromTable(temp_output, filter_columns, output));
  RETURN_CYLON_STATUS_IF_FAILED(SetArrowIndexForLocResultTable(index, s_index, e_index, output, indexing_type_));
  return cylon::Status::OK();
}
cylon::Status cylon::ArrowLocIndexer::loc(const std::shared_ptr<arrow::Scalar> &start_index,
										  const std::shared_ptr<arrow::Scalar> &end_index,
										  const int start_column_index,
										  const int end_column_index,
										  const std::shared_ptr<Table> &input_table,
										  std::shared_ptr<cylon::Table> &output) {
  auto index = input_table->GetArrowIndex();
  std::shared_ptr<cylon::Table> temp_output;
  int64_t s_index = -1;
  int64_t e_index = -1;
  RETURN_CYLON_STATUS_IF_FAILED(GetArrowLocFilterIndices(start_index, end_index, index, &s_index, &e_index));
  // filter columns include both boundaries
  std::vector<int> filter_columns;
  RETURN_CYLON_STATUS_IF_FAILED(GetColumnIndicesFromLimits(start_column_index, end_column_index, filter_columns));
  RETURN_CYLON_STATUS_IF_FAILED(SliceTableByRange(s_index, e_index, input_table, temp_output));
  RETURN_CYLON_STATUS_IF_FAILED(FilterColumnsFromTable(temp_output, filter_columns, output));
  RETURN_CYLON_STATUS_IF_FAILED(SetArrowIndexForLocResultTable(index, s_index, e_index, output, indexing_type_));
  return cylon::Status::OK();
}
cylon::Status cylon::ArrowLocIndexer::loc(const std::shared_ptr<arrow::Scalar> &start_index,
										  const std::shared_ptr<arrow::Scalar> &end_index,
										  const std::vector<int> &columns,
										  const std::shared_ptr<Table> &input_table,
										  std::shared_ptr<cylon::Table> &output) {
  std::shared_ptr<cylon::Table> temp_output;
  int64_t s_index = -1;
  int64_t e_index = -1;
  auto index = input_table->GetArrowIndex();
  RETURN_CYLON_STATUS_IF_FAILED(GetArrowLocFilterIndices(start_index, end_index, index, &s_index, &e_index));
  RETURN_CYLON_STATUS_IF_FAILED(SliceTableByRange(s_index, e_index, input_table, temp_output));
  RETURN_CYLON_STATUS_IF_FAILED(FilterColumnsFromTable(temp_output, columns, output));
  RETURN_CYLON_STATUS_IF_FAILED(SetArrowIndexForLocResultTable(index, s_index, e_index, output, indexing_type_));
  return cylon::Status::OK();
}
cylon::Status cylon::ArrowLocIndexer::loc(const std::shared_ptr<arrow::Array> &indices,
										  const int column_index,
										  const std::shared_ptr<Table> &input_table,
										  std::shared_ptr<cylon::Table> &output) {
  std::vector<int64_t> filter_indices;
  std::shared_ptr<cylon::Table> temp_table;
  auto index = input_table->GetArrowIndex();
  RETURN_CYLON_STATUS_IF_FAILED(ResolveArrowLocIndices(indices, index, filter_indices));
  RETURN_CYLON_STATUS_IF_FAILED(GetTableFromIndices(input_table, filter_indices, temp_table));
  std::vector<int> columns = {column_index};
  RETURN_CYLON_STATUS_IF_FAILED(FilterColumnsFromTable(temp_table, columns, output));
  RETURN_CYLON_STATUS_IF_FAILED(SetArrowIndexForLocResultTable(index, filter_indices, output, indexing_type_));
  return cylon::Status::OK();
}
cylon::Status cylon::ArrowLocIndexer::loc(const std::shared_ptr<arrow::Array> &indices,
										  const int start_column,
										  const int end_column,
										  const std::shared_ptr<Table> &input_table,
										  std::shared_ptr<cylon::Table> &output) {
  std::vector<int64_t> filter_indices;
  std::shared_ptr<cylon::Table> temp_table;
  auto index = input_table->GetArrowIndex();
  RETURN_CYLON_STATUS_IF_FAILED(ResolveArrowLocIndices(indices, index, filter_indices));
  RETURN_CYLON_STATUS_IF_FAILED(GetTableFromIndices(input_table, filter_indices, temp_table));
  std::vector<int> filter_columns;
  RETURN_CYLON_STATUS_IF_FAILED(GetColumnIndicesFromLimits(start_column, end_column, filter_columns));
  RETURN_CYLON_STATUS_IF_FAILED(FilterColumnsFromTable(temp_table, filter_columns, output));
  RETURN_CYLON_STATUS_IF_FAILED(SetArrowIndexForLocResultTable(index, filter_indices, output, indexing_type_));
  return cylon::Status::OK();
}
cylon::Status cylon::ArrowLocIndexer::loc(const std::shared_ptr<arrow::Array> &indices,
										  const std::vector<int> &columns,
										  const std::shared_ptr<Table> &input_table,
										  std::shared_ptr<cylon::Table> &output) {
  std::vector<int64_t> filter_indices;
  std::shared_ptr<cylon::Table> temp_table;
  auto index = input_table->GetArrowIndex();
  RETURN_CYLON_STATUS_IF_FAILED(ResolveArrowLocIndices(indices, index, filter_indices));
  RETURN_CYLON_STATUS_IF_FAILED(GetTableFromIndices(input_table, filter_indices, temp_table));
  RETURN_CYLON_STATUS_IF_FAILED(FilterColumnsFromTable(temp_table, columns, output));
  RETURN_CYLON_STATUS_IF_FAILED(SetArrowIndexForLocResultTable(index, filter_indices, output, indexing_type_));
  return cylon::Status::OK();
}
cylon::IndexingType cylon::ArrowLocIndexer::GetIndexingType() {
  return indexing_type_;
}

cylon::ArrowILocIndexer::ArrowILocIndexer(cylon::IndexingType indexing_type)
	: ArrowLocIndexer(indexing_type) {}
cylon::Status cylon::ArrowILocIndexer::loc(const std::shared_ptr<arrow::Scalar> &start_index,
										   const std::shared_ptr<arrow::Scalar> &end_index,
										   const int column_index,
										   const std::shared_ptr<Table> &input_table,
										   std::shared_ptr<cylon::Table> &output) {
  std::shared_ptr<cylon::BaseArrowIndex> range_index;
  std::shared_ptr<cylon::Table> temp_out;
  std::shared_ptr<arrow::Int64Scalar> start_index_scalar = std::static_pointer_cast<arrow::Int64Scalar>(start_index);
  std::shared_ptr<arrow::Int64Scalar> end_index_scalar = std::static_pointer_cast<arrow::Int64Scalar>(end_index);
  int64_t start_index_pos = start_index_scalar->value;
  int64_t end_index_pos = end_index_scalar->value;
  RETURN_CYLON_STATUS_IF_FAILED(SliceTableByRange(start_index_pos, end_index_pos, input_table, temp_out));
  std::vector<int> filter_columns = {column_index};
  RETURN_CYLON_STATUS_IF_FAILED(FilterColumnsFromTable(temp_out, filter_columns, output));
  // default of ILocIndex based operation is a range index
  RETURN_CYLON_STATUS_IF_FAILED(cylon::IndexUtil::BuildArrowRangeIndex(output, range_index));
  RETURN_CYLON_STATUS_IF_FAILED(output->SetArrowIndex(range_index, false));
  return Status::OK();
}
cylon::Status cylon::ArrowILocIndexer::loc(const std::shared_ptr<arrow::Scalar> &start_index,
										   const std::shared_ptr<arrow::Scalar> &end_index,
										   const int start_column_index,
										   const int end_column_index,
										   const std::shared_ptr<Table> &input_table,
										   std::shared_ptr<cylon::Table> &output) {
  std::shared_ptr<cylon::BaseArrowIndex> range_index;
  std::shared_ptr<cylon::Table> temp_out;
  std::shared_ptr<arrow::Int64Scalar> start_index_scalar = std::static_pointer_cast<arrow::Int64Scalar>(start_index);
  std::shared_ptr<arrow::Int64Scalar> end_index_scalar = std::static_pointer_cast<arrow::Int64Scalar>(end_index);
  int64_t start_index_pos = start_index_scalar->value;
  int64_t end_index_pos = end_index_scalar->value;
  RETURN_CYLON_STATUS_IF_FAILED(SliceTableByRange(start_index_pos, end_index_pos, input_table, temp_out));
  // filter columns include both boundaries
  std::vector<int> filter_columns;
  RETURN_CYLON_STATUS_IF_FAILED(GetColumnIndicesFromLimits(start_column_index, end_column_index, filter_columns));
  RETURN_CYLON_STATUS_IF_FAILED(FilterColumnsFromTable(temp_out, filter_columns, output));
  // default of ILocIndex based operation is a range index
  RETURN_CYLON_STATUS_IF_FAILED(cylon::IndexUtil::BuildArrowRangeIndex(output, range_index));
  RETURN_CYLON_STATUS_IF_FAILED(output->SetArrowIndex(range_index, false));
  return Status::OK();
}
cylon::Status cylon::ArrowILocIndexer::loc(const std::shared_ptr<arrow::Scalar> &start_index,
										   const std::shared_ptr<arrow::Scalar> &end_index,
										   const std::vector<int> &columns,
										   const std::shared_ptr<Table> &input_table,
										   std::shared_ptr<cylon::Table> &output) {
  std::shared_ptr<cylon::BaseArrowIndex> range_index;
  std::shared_ptr<cylon::Table> temp_out;
  std::shared_ptr<arrow::Int64Scalar> start_index_scalar = std::static_pointer_cast<arrow::Int64Scalar>(start_index);
  std::shared_ptr<arrow::Int64Scalar> end_index_scalar = std::static_pointer_cast<arrow::Int64Scalar>(end_index);
  int64_t start_index_pos = start_index_scalar->value;
  int64_t end_index_pos = end_index_scalar->value;
  RETURN_CYLON_STATUS_IF_FAILED(SliceTableByRange(start_index_pos, end_index_pos, input_table, temp_out));
  RETURN_CYLON_STATUS_IF_FAILED(FilterColumnsFromTable(temp_out, columns, output));
  // default of ILocIndex based operation is a range index
  RETURN_CYLON_STATUS_IF_FAILED(cylon::IndexUtil::BuildArrowRangeIndex(output, range_index));
  RETURN_CYLON_STATUS_IF_FAILED(output->SetArrowIndex(range_index, false));
  return Status::OK();
}
cylon::Status cylon::ArrowILocIndexer::loc(const std::shared_ptr<arrow::Array> &indices,
										   const int column_index,
										   const std::shared_ptr<Table> &input_table,
										   std::shared_ptr<cylon::Table> &output) {
  std::shared_ptr<cylon::BaseArrowIndex> range_index;
  std::vector<int64_t> i_indices(indices->length());
  std::shared_ptr<cylon::Table> temp_out;
  RETURN_CYLON_STATUS_IF_FAILED(ResolveArrowILocIndices(indices, i_indices));
  RETURN_CYLON_STATUS_IF_FAILED(GetTableFromIndices(input_table, i_indices, temp_out));
  std::vector<int> columns{column_index};
  RETURN_CYLON_STATUS_IF_FAILED(FilterColumnsFromTable(temp_out, columns, output));
  // default of ILocIndex based operation is a range index
  RETURN_CYLON_STATUS_IF_FAILED(cylon::IndexUtil::BuildArrowRangeIndex(output, range_index));
  RETURN_CYLON_STATUS_IF_FAILED(output->SetArrowIndex(range_index, false));
  return Status::OK();
}
cylon::Status cylon::ArrowILocIndexer::loc(const std::shared_ptr<arrow::Array> &indices,
										   const int start_column,
										   const int end_column,
										   const std::shared_ptr<Table> &input_table,
										   std::shared_ptr<cylon::Table> &output) {
  std::shared_ptr<cylon::BaseArrowIndex> range_index;
  std::vector<int64_t> i_indices(indices->length());
  std::shared_ptr<cylon::Table> temp_out;
  RETURN_CYLON_STATUS_IF_FAILED(ResolveArrowILocIndices(indices, i_indices));
  RETURN_CYLON_STATUS_IF_FAILED(GetTableFromIndices(input_table, i_indices, temp_out));
  std::vector<int> filter_columns;
  RETURN_CYLON_STATUS_IF_FAILED(GetColumnIndicesFromLimits(start_column, end_column, filter_columns));
  RETURN_CYLON_STATUS_IF_FAILED(FilterColumnsFromTable(temp_out, filter_columns, output));
  // default of ILocIndex based operation is a range index
  RETURN_CYLON_STATUS_IF_FAILED(cylon::IndexUtil::BuildArrowRangeIndex(output, range_index));
  RETURN_CYLON_STATUS_IF_FAILED(output->SetArrowIndex(range_index, false));
  return Status::OK();
}
cylon::Status cylon::ArrowILocIndexer::loc(const std::shared_ptr<arrow::Array> &indices,
										   const std::vector<int> &columns,
										   const std::shared_ptr<Table> &input_table,
										   std::shared_ptr<cylon::Table> &output) {
  std::shared_ptr<cylon::BaseArrowIndex> range_index;
  std::vector<int64_t> i_indices(indices->length());
  std::shared_ptr<cylon::Table> temp_out;
  RETURN_CYLON_STATUS_IF_FAILED(ResolveArrowILocIndices(indices, i_indices));
  RETURN_CYLON_STATUS_IF_FAILED(GetTableFromIndices(input_table, i_indices, temp_out));
  RETURN_CYLON_STATUS_IF_FAILED(FilterColumnsFromTable(temp_out, columns, output));
  // default of ILocIndex based operation is a range index
  RETURN_CYLON_STATUS_IF_FAILED(cylon::IndexUtil::BuildArrowRangeIndex(output, range_index));
  RETURN_CYLON_STATUS_IF_FAILED(output->SetArrowIndex(range_index, false));
  return Status::OK();
}
cylon::IndexingType cylon::ArrowILocIndexer::GetIndexingType() {
  return ArrowLocIndexer::GetIndexingType();
}

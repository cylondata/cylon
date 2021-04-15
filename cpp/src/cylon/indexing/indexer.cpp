
#include "indexer.hpp"
#include "index_utils.hpp"

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
  auto ctx = output->GetContext();
  auto const pool = cylon::ToArrowPool(ctx);
  auto const index_arr = index->GetIndexArray();
  arrow::Int64Builder builder(pool);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.AppendValues(sub_index_locations));
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.Finish(&sub_index_pos_arr));
  arrow::Result<arrow::Datum> datum = arrow::compute::Take(index_arr, sub_index_pos_arr);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(datum.status());
  sub_index_arr = datum.ValueOrDie().make_array();
  RETURN_CYLON_STATUS_IF_FAILED(BuildArrowIndexFromArrayByKernel(indexing_type, sub_index_arr, pool, loc_index));
  RETURN_CYLON_STATUS_IF_FAILED(output->Set_ArrowIndex(loc_index, false));
  return cylon::Status::OK();
}

cylon::Status SetArrowIndexForLocResultTable(const std::shared_ptr<cylon::BaseArrowIndex> &index,
											 int64_t &start_pos,
											 int64_t &end_pos,
											 std::shared_ptr<cylon::Table> &output,
											 cylon::IndexingType indexing_type) {
  std::shared_ptr<cylon::BaseArrowIndex> loc_index;
  std::shared_ptr<arrow::Array> sub_index_arr;
  auto ctx = output->GetContext();
  auto const pool = cylon::ToArrowPool(ctx);
  auto const index_arr = index->GetIndexArray();
  sub_index_arr = index_arr->Slice(start_pos, (end_pos - start_pos + 1));
  RETURN_CYLON_STATUS_IF_FAILED(BuildArrowIndexFromArrayByKernel(indexing_type, sub_index_arr, pool, loc_index));
  RETURN_CYLON_STATUS_IF_FAILED(output->Set_ArrowIndex(loc_index, false));  return cylon::Status::OK();

}

cylon::Status GetArrowLocFilterIndices(const std::shared_ptr<arrow::Scalar> &start_index,
									   const std::shared_ptr<arrow::Scalar> &end_index,
									   const std::shared_ptr<cylon::BaseArrowIndex> &index,
									   int64_t &s_index,
									   int64_t &e_index) {
  std::shared_ptr<arrow::Table> out_artb;
  RETURN_CYLON_STATUS_IF_FAILED(cylon::CheckIsIndexValueUnique(start_index, index));
  RETURN_CYLON_STATUS_IF_FAILED(cylon::CheckIsIndexValueUnique(end_index, index));
  RETURN_CYLON_STATUS_IF_FAILED(index->LocationByValue(start_index, s_index));
  RETURN_CYLON_STATUS_IF_FAILED(index->LocationByValue(end_index, e_index));
  return cylon::Status::OK();
}

cylon::Status SliceTableByRange(const int64_t start_index,
								const int64_t end_index,
								const std::shared_ptr<cylon::Table> &input_table,
								std::shared_ptr<cylon::Table> &output) {

  std::shared_ptr<arrow::Table> out_artb;
  auto artb = input_table->get_table();
  auto ctx = input_table->GetContext();
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
  auto ctx = input_table->GetContext();
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
	output_indices.push_back(val);
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
  auto ctx = input_table->GetContext();
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
  auto ctx = input_table->GetContext();
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


cylon::Status cylon::CheckIsIndexValueUnique(const std::shared_ptr<arrow::Scalar> &index_value,
											 const std::shared_ptr<BaseArrowIndex> &index) {

  if (index->GetIndexingType() == cylon::IndexingType::Range) {
	return cylon::Status::OK();
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
		return cylon::Status(cylon::Code::IndexError, "Index values are not unique.");
	  }
	}
	return cylon::Status::OK();
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
  RETURN_CYLON_STATUS_IF_FAILED(GetArrowLocFilterIndices(start_index, end_index, index, s_index, e_index));
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
  Status status_build;
  auto index = input_table->GetArrowIndex();
  std::shared_ptr<cylon::Table> temp_output;
  int64_t s_index, e_index = -1;

  status_build = GetArrowLocFilterIndices(start_index, end_index, index, s_index, e_index);

  if (!status_build.is_ok()) {
	return status_build;
  }

  // filter columns include both boundaries
  std::vector<int> filter_columns;

  status_build = GetColumnIndicesFromLimits(start_column_index, end_column_index, filter_columns);

  if (!status_build.is_ok()) {
	LOG(ERROR) << "Error occurred building column indices from boundaries";
	return status_build;
  }

  status_build = SliceTableByRange(s_index, e_index, input_table, temp_output);

  if (!status_build.is_ok()) {
	LOG(ERROR) << "Error occurred in filtering indices from table";
	return status_build;
  }

  status_build = FilterColumnsFromTable(temp_output, filter_columns, output);

  if (!status_build.is_ok()) {
	LOG(ERROR) << "Error occurred in filtering columns from table";
	return status_build;
  }

  status_build = SetArrowIndexForLocResultTable(index, s_index, e_index, output, indexing_type_);

  if (!status_build.is_ok()) {
	LOG(ERROR) << "Error occurred in setting index for output table";
	return status_build;
  }

  return cylon::Status::OK();
}
cylon::Status cylon::ArrowLocIndexer::loc(const std::shared_ptr<arrow::Scalar> &start_index,
										  const std::shared_ptr<arrow::Scalar> &end_index,
										  const std::vector<int> &columns,
										  const std::shared_ptr<Table> &input_table,
										  std::shared_ptr<cylon::Table> &output) {
  Status status_build;
  std::shared_ptr<cylon::Table> temp_output;
  int64_t s_index, e_index = -1;
  auto index = input_table->GetArrowIndex();
  status_build = GetArrowLocFilterIndices(start_index, end_index, index, s_index, e_index);

  if (!status_build.is_ok()) {
	return status_build;
  }

  status_build = SliceTableByRange(s_index, e_index, input_table, temp_output);

  if (!status_build.is_ok()) {
	LOG(ERROR) << "Error occurred in filtering indices from table";
	return status_build;
  }

  status_build = FilterColumnsFromTable(temp_output, columns, output);

  if (!status_build.is_ok()) {
	LOG(ERROR) << "Error occurred in filtering columns from table";
	return status_build;
  }

  status_build = SetArrowIndexForLocResultTable(index, s_index, e_index, output, indexing_type_);

  if (!status_build.is_ok()) {
	LOG(ERROR) << "Error occurred in setting index for output table";
	return status_build;
  }

  return cylon::Status::OK();
}
cylon::Status cylon::ArrowLocIndexer::loc(const std::shared_ptr<arrow::Array> &indices,
										  const int column_index,
										  const std::shared_ptr<Table> &input_table,
										  std::shared_ptr<cylon::Table> &output) {
  Status status;
  std::vector<int64_t> filter_indices;
  std::shared_ptr<cylon::Table> temp_table;
  auto index = input_table->GetArrowIndex();
  status = ResolveArrowLocIndices(indices, index, filter_indices);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in resolving indices for table filtering";
	return status;
  }

  status = GetTableFromIndices(input_table, filter_indices, temp_table);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table from filter indices";
	return status;
  }

  std::vector<int> columns = {column_index};

  status = FilterColumnsFromTable(temp_table, columns, output);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table from selected columns";
	return status;
  }

  status = SetArrowIndexForLocResultTable(index, filter_indices, output, indexing_type_);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in setting index for output table";
	return status;
  }

  return cylon::Status::OK();
}
cylon::Status cylon::ArrowLocIndexer::loc(const std::shared_ptr<arrow::Array> &indices,
										  const int start_column,
										  const int end_column,
										  const std::shared_ptr<Table> &input_table,
										  std::shared_ptr<cylon::Table> &output) {
  Status status;
  std::vector<int64_t> filter_indices;
  std::shared_ptr<cylon::Table> temp_table;
  auto index = input_table->GetArrowIndex();
  status = ResolveArrowLocIndices(indices, index, filter_indices);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in resolving indices for table filtering";
	return status;
  }

  status = GetTableFromIndices(input_table, filter_indices, temp_table);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table from filter indices";
	return status;
  }

  std::vector<int> filter_columns;

  status = GetColumnIndicesFromLimits(start_column, end_column, filter_columns);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in getting column indices from boundaries";
	return status;
  }

  status = FilterColumnsFromTable(temp_table, filter_columns, output);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table from selected columns";
	return status;
  }

  status = SetArrowIndexForLocResultTable(index, filter_indices, output, indexing_type_);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in setting index for output table";
	return status;
  }

  return cylon::Status::OK();
}
cylon::Status cylon::ArrowLocIndexer::loc(const std::shared_ptr<arrow::Array> &indices,
										  const std::vector<int> &columns,
										  const std::shared_ptr<Table> &input_table,
										  std::shared_ptr<cylon::Table> &output) {
  Status status;
  std::vector<int64_t> filter_indices;
  std::shared_ptr<cylon::Table> temp_table;
  auto index = input_table->GetArrowIndex();
  status = ResolveArrowLocIndices(indices, index, filter_indices);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in resolving indices for table filtering";
	return status;
  }

  status = GetTableFromIndices(input_table, filter_indices, temp_table);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table from filter indices";
	return status;
  }

  status = FilterColumnsFromTable(temp_table, columns, output);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table from selected columns";
	return status;
  }

  status = SetArrowIndexForLocResultTable(index, filter_indices, output, indexing_type_);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in setting index for output table";
	return status;
  }

  return cylon::Status::OK();
}
cylon::IndexingType cylon::ArrowLocIndexer::GetIndexingType() {
  return indexing_type_;
}

cylon::ArrowILocIndexer::ArrowILocIndexer(cylon::IndexingType indexing_type)
	: ArrowLocIndexer(indexing_type), indexing_type_(indexing_type) {}
cylon::Status cylon::ArrowILocIndexer::loc(const std::shared_ptr<arrow::Scalar> &start_index,
										   const std::shared_ptr<arrow::Scalar> &end_index,
										   const int column_index,
										   const std::shared_ptr<Table> &input_table,
										   std::shared_ptr<cylon::Table> &output) {
  cylon::Status status;
  std::shared_ptr<cylon::BaseArrowIndex> range_index;
  std::shared_ptr<cylon::Table> temp_out;
  std::shared_ptr<arrow::Int64Scalar>
	  start_index_scalar = std::static_pointer_cast<arrow::TypeTraits<arrow::Int64Type>::ScalarType>(start_index);
  std::shared_ptr<arrow::Int64Scalar>
	  end_index_scalar = std::static_pointer_cast<arrow::TypeTraits<arrow::Int64Type>::ScalarType>(end_index);
  int64_t start_index_pos = start_index_scalar->value; //*((int64_t *)start_index);
  int64_t end_index_pos = end_index_scalar->value; //*((int64_t *)end_index);

  status = SliceTableByRange(start_index_pos, end_index_pos, input_table, temp_out);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in filtering table based on index range";
	return status;
  }
  std::vector<int> filter_columns = {column_index};

  status = FilterColumnsFromTable(temp_out, filter_columns, output);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in filtering table based on index range";
	return status;
  }
  // default of ILocIndex based operation is a range index
  status = cylon::IndexUtil::BuildArrowRangeIndex(output, range_index);

  output->Set_ArrowIndex(range_index, false);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating range index for output table";
	return status;
  }

  return Status::OK();
}
cylon::Status cylon::ArrowILocIndexer::loc(const std::shared_ptr<arrow::Scalar> &start_index,
										   const std::shared_ptr<arrow::Scalar> &end_index,
										   const int start_column_index,
										   const int end_column_index,
										   const std::shared_ptr<Table> &input_table,
										   std::shared_ptr<cylon::Table> &output) {
  cylon::Status status;
  std::shared_ptr<cylon::BaseArrowIndex> range_index;
  std::shared_ptr<cylon::Table> temp_out;
  std::shared_ptr<arrow::Int64Scalar>
	  start_index_scalar = std::static_pointer_cast<arrow::TypeTraits<arrow::Int64Type>::ScalarType>(start_index);
  std::shared_ptr<arrow::Int64Scalar>
	  end_index_scalar = std::static_pointer_cast<arrow::TypeTraits<arrow::Int64Type>::ScalarType>(end_index);
  int64_t start_index_pos = start_index_scalar->value; //*((int64_t *)start_index);
  int64_t end_index_pos = end_index_scalar->value; //*((int64_t *)end_index);

  status = SliceTableByRange(start_index_pos, end_index_pos, input_table, temp_out);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in filtering table based on index range";
	return status;
  }

  // filter columns include both boundaries
  std::vector<int> filter_columns;

  status = GetColumnIndicesFromLimits(start_column_index, end_column_index, filter_columns);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating column range based column filter";
	return status;
  }

  status = FilterColumnsFromTable(temp_out, filter_columns, output);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in filtering table based on index range";
	return status;
  }
  // default of ILocIndex based operation is a range index
  status = cylon::IndexUtil::BuildArrowRangeIndex(output, range_index);

  output->Set_ArrowIndex(range_index, false);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating range index for output table";
	return status;
  }

  return Status::OK();
}
cylon::Status cylon::ArrowILocIndexer::loc(const std::shared_ptr<arrow::Scalar> &start_index,
										   const std::shared_ptr<arrow::Scalar> &end_index,
										   const std::vector<int> &columns,
										   const std::shared_ptr<Table> &input_table,
										   std::shared_ptr<cylon::Table> &output) {
  cylon::Status status;
  std::shared_ptr<cylon::BaseArrowIndex> range_index;
  std::shared_ptr<cylon::Table> temp_out;
  std::shared_ptr<arrow::Int64Scalar>
	  start_index_scalar = std::static_pointer_cast<arrow::TypeTraits<arrow::Int64Type>::ScalarType>(start_index);
  std::shared_ptr<arrow::Int64Scalar>
	  end_index_scalar = std::static_pointer_cast<arrow::TypeTraits<arrow::Int64Type>::ScalarType>(end_index);
  int64_t start_index_pos = start_index_scalar->value; //*((int64_t *)start_index);
  int64_t end_index_pos = end_index_scalar->value; //*((int64_t *)end_index);

  status = SliceTableByRange(start_index_pos, end_index_pos, input_table, temp_out);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in filtering table based on index range";
	return status;
  }

  status = FilterColumnsFromTable(temp_out, columns, output);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in filtering table based on columns";
	return status;
  }
  // default of ILocIndex based operation is a range index
  status = cylon::IndexUtil::BuildArrowRangeIndex(output, range_index);

  output->Set_ArrowIndex(range_index, false);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating range index for output table";
	return status;
  }

  return Status::OK();
}
cylon::Status cylon::ArrowILocIndexer::loc(const std::shared_ptr<arrow::Array> &indices,
										   const int column_index,
										   const std::shared_ptr<Table> &input_table,
										   std::shared_ptr<cylon::Table> &output) {
  Status status;
  std::shared_ptr<cylon::BaseArrowIndex> range_index;
  std::vector<int64_t> i_indices;
  std::shared_ptr<cylon::Table> temp_out;
  status = ResolveArrowILocIndices(indices, i_indices);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in iloc index resolving";
  }

  status = GetTableFromIndices(input_table, i_indices, temp_out);

  if (!status.is_ok()) {
	LOG(ERROR) << "Filtering table from indices failed!";
	return status;
  }

  std::vector<int> columns{column_index};

  status = FilterColumnsFromTable(temp_out, columns, output);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in filtering table based on index range";
	return status;
  }
  // default of ILocIndex based operation is a range index
  status = cylon::IndexUtil::BuildArrowRangeIndex(output, range_index);

  output->Set_ArrowIndex(range_index, false);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating range index for output table";
	return status;
  }

  return Status::OK();
}
cylon::Status cylon::ArrowILocIndexer::loc(const std::shared_ptr<arrow::Array> &indices,
										   const int start_column,
										   const int end_column,
										   const std::shared_ptr<Table> &input_table,
										   std::shared_ptr<cylon::Table> &output) {
  Status status;
  std::shared_ptr<cylon::BaseArrowIndex> range_index;
  std::vector<int64_t> i_indices;
  std::shared_ptr<cylon::Table> temp_out;
  status = ResolveArrowILocIndices(indices, i_indices);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in iloc index resolving";
  }

  status = GetTableFromIndices(input_table, i_indices, temp_out);

  if (!status.is_ok()) {
	LOG(ERROR) << "Filtering table from indices failed!";
	return status;
  }

  std::vector<int> filter_columns;

  status = GetColumnIndicesFromLimits(start_column, end_column, filter_columns);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating column range based column filter";
	return status;
  }

  status = FilterColumnsFromTable(temp_out, filter_columns, output);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in filtering table based on index range";
	return status;
  }
  // default of ILocIndex based operation is a range index
  status = cylon::IndexUtil::BuildArrowRangeIndex(output, range_index);

  output->Set_ArrowIndex(range_index, false);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating range index for output table";
	return status;
  }

  return Status::OK();
}
cylon::Status cylon::ArrowILocIndexer::loc(const std::shared_ptr<arrow::Array> &indices,
										   const std::vector<int> &columns,
										   const std::shared_ptr<Table> &input_table,
										   std::shared_ptr<cylon::Table> &output) {
  Status status;
  std::shared_ptr<cylon::BaseArrowIndex> range_index;
  std::vector<int64_t> i_indices;
  std::shared_ptr<cylon::Table> temp_out;
  status = ResolveArrowILocIndices(indices, i_indices);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in iloc index resolving";
  }

  status = GetTableFromIndices(input_table, i_indices, temp_out);

  if (!status.is_ok()) {
	LOG(ERROR) << "Filtering table from indices failed!";
	return status;
  }

  status = FilterColumnsFromTable(temp_out, columns, output);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in filtering table based on index range";
	return status;
  }
  // default of ILocIndex based operation is a range index
  status = cylon::IndexUtil::BuildArrowRangeIndex(output, range_index);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating range index for output table";
	return status;
  }

  output->Set_ArrowIndex(range_index, false);

  return Status::OK();
}
cylon::IndexingType cylon::ArrowILocIndexer::GetIndexingType() {
  return ArrowLocIndexer::GetIndexingType();
}

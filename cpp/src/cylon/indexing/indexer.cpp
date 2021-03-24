
#include "indexer.hpp"
#include "index_utils.hpp"

bool IsValidColumnRange(int &column_id, std::shared_ptr<cylon::Table> &input_table) {
  return (column_id < input_table->Rows() and column_id >= 0);
}

bool IsRangeIndex(std::shared_ptr<cylon::BaseIndex> &index) {
  if (std::shared_ptr<cylon::RangeIndex> r = std::dynamic_pointer_cast<cylon::RangeIndex>(index)) {
    return true;
  }
  return false;
}

cylon::Status BuildIndexFromArrayByKernel(cylon::IndexingSchema indexing_schema,
                                          std::shared_ptr<arrow::Array> &sub_index_arr,
                                          arrow::MemoryPool *pool,
                                          std::shared_ptr<cylon::BaseIndex> &loc_index) {
  if (indexing_schema == cylon::IndexingSchema::Hash) {
    return cylon::IndexUtil::BuildHashIndexFromArray(sub_index_arr, pool, loc_index);
  } else if (indexing_schema == cylon::IndexingSchema::Linear) {
    return cylon::IndexUtil::BuildLinearIndexFromArray(sub_index_arr, pool, loc_index);
  } else if (indexing_schema == cylon::IndexingSchema::Range) {
    return cylon::IndexUtil::BuildRangeIndexFromArray(sub_index_arr, pool, loc_index);
  } else if (indexing_schema == cylon::IndexingSchema::BinaryTree) {
    return cylon::Status(cylon::Code::NotImplemented, "Binary Tree Indexing not implemented!");
  } else if (indexing_schema == cylon::IndexingSchema::BTree) {
    return cylon::Status(cylon::Code::NotImplemented, "B-Tree Indexing not implemented!");
  } else {
    return cylon::Status(cylon::Code::TypeError, "Unknown indexing scheme.");
  }
}

cylon::Status SetIndexForLocResultTable(const std::shared_ptr<cylon::BaseIndex> &index,
                                        std::vector<int64_t> &sub_index_locations,
                                        std::shared_ptr<cylon::Table> &output,
                                        cylon::IndexingSchema indexing_schema) {

  std::shared_ptr<cylon::BaseIndex> loc_index;
  std::shared_ptr<arrow::Array> sub_index_pos_arr;
  std::shared_ptr<arrow::Array> sub_index_arr;
  arrow::Status status;
  cylon::Status cylon_status;

  auto ctx = output->GetContext();
  auto pool = cylon::ToArrowPool(ctx);

  //LOG(INFO) << "Set Index for location output with Non-RangeIndex";
  auto index_arr = index->GetIndexArray();
  arrow::Int64Builder builder(pool);
  status = builder.AppendValues(sub_index_locations);

  if (!status.ok()) {
    LOG(ERROR) << "HashIndex array builder append failed!";
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(status);
  }

  status = builder.Finish(&sub_index_pos_arr);

  if (!status.ok()) {
    LOG(ERROR) << "HashIndex array builder finish failed!";
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(status);
  }

  arrow::Result<arrow::Datum> datum = arrow::compute::Take(index_arr, sub_index_pos_arr);

  if (!datum.status().ok()) {
    LOG(ERROR) << "Sub HashIndex array creation failed!";
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(datum.status());
  }

  sub_index_arr = datum.ValueOrDie().make_array();

  cylon_status = BuildIndexFromArrayByKernel(indexing_schema, sub_index_arr, pool, loc_index);

  if (!cylon_status.is_ok()) {
    LOG(ERROR) << "Error occurred in resolving kernel for index array building";
    return cylon_status;
  }

  output->Set_Index(loc_index, false);
  return cylon::Status::OK();
}

cylon::Status SetIndexForLocResultTable(const std::shared_ptr<cylon::BaseIndex> &index,
                                        int64_t &start_pos,
                                        int64_t &end_pos,
                                        std::shared_ptr<cylon::Table> &output,
                                        cylon::IndexingSchema indexing_schema) {
  std::shared_ptr<cylon::BaseIndex> loc_index;
  std::shared_ptr<arrow::Array> sub_index_arr;
  auto ctx = output->GetContext();
  auto pool = cylon::ToArrowPool(ctx);

  LOG(INFO) << "Set Index for location output with Non-RangeIndex";
  auto index_arr = index->GetIndexArray();
  sub_index_arr = index_arr->Slice(start_pos, (end_pos - start_pos + 1));
  BuildIndexFromArrayByKernel(indexing_schema, sub_index_arr, pool, loc_index);

  output->Set_Index(loc_index, false);

  return cylon::Status::OK();
}

cylon::Status GetLocFilterIndices(const void *start_index,
                                  const void *end_index,
                                  const std::shared_ptr<cylon::BaseIndex> &index,
                                  int64_t &s_index,
                                  int64_t &e_index) {
  cylon::Status status1, status2, status_build;
  std::shared_ptr<arrow::Table> out_artb;
  bool is_index_unique;
  status_build = cylon::CheckIsIndexValueUnique(start_index, index, is_index_unique);

  if (!status_build.is_ok()) {
    std::string error_msg = "Error occurred in checking uniqueness of index value";
    LOG(ERROR) << error_msg;
    return cylon::Status(cylon::Code::IndexError, error_msg);
  }

  if (!is_index_unique) {
    LOG(ERROR) << "Index value must be unique";
    return cylon::Status(cylon::Code::KeyError);
  }

  cylon::CheckIsIndexValueUnique(end_index, index, is_index_unique);

  if (!is_index_unique) {
    LOG(ERROR) << "Index value must be unique";
    return cylon::Status(cylon::Code::KeyError);
  }

  status1 = index->LocationByValue(start_index, s_index);
  status2 = index->LocationByValue(end_index, e_index);

  if (!(status1.is_ok() and status2.is_ok())) {
    LOG(ERROR) << "Error occurred in extracting indices!";
    return cylon::Status(cylon::Code::IndexError);
  }

  return cylon::Status::OK();
}

cylon::Status SliceTableByRange(const int64_t start_index,
                                const int64_t end_index,
                                const std::shared_ptr<cylon::Table> &input_table,
                                std::shared_ptr<cylon::Table> &output) {

  cylon::Status status_build;
  std::shared_ptr<arrow::Table> out_artb;

  auto artb = input_table->get_table();
  auto ctx = input_table->GetContext();
  // + 1 added include the end boundary
  out_artb = artb->Slice(start_index, (end_index - start_index + 1));

  status_build = cylon::Table::FromArrowTable(ctx, out_artb, output);

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error occurred in creating loc output table";
    return status_build;
  }

  return cylon::Status::OK();
}

cylon::Status GetColumnIndicesFromLimits(const int &start_column, const int &end_column, std::vector<int> &selected_columns) {

  if (start_column > end_column) {
    LOG(ERROR) << "Invalid column boundaries";
    return cylon::Status(cylon::Code::Invalid);
  }

  for (int s = start_column; s <= end_column; s++) {
    selected_columns.push_back(s);
  }

  return cylon::Status::OK();
}

cylon::Status FilterColumnsFromTable(const std::shared_ptr<cylon::Table> &input_table,
                                     const std::vector<int> &filter_columns,
                                     std::shared_ptr<cylon::Table> &output) {

  cylon::Status status_build;
  auto ctx = input_table->GetContext();
  arrow::Result<std::shared_ptr<arrow::Table>> result = input_table->get_table()->SelectColumns(filter_columns);

  if (!result.status().ok()) {
    LOG(ERROR) << "Column selection failed in loc operation!";
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
  }

  status_build = cylon::Table::FromArrowTable(ctx, result.ValueOrDie(), output);

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error occurred in creating Cylon Table from Arrow Table";
    return status_build;
  }

  return cylon::Status::OK();
}

static cylon::Status ResolveILocIndices(const std::vector<void *> &input_indices,
                                        std::vector<int64_t> &output_indices) {
  cylon::Status status;
  for (size_t ix = 0; ix < input_indices.size(); ix++) {
    int64_t val = *static_cast<int64_t *>(input_indices.at(ix));
    output_indices.push_back(val);
  }
  return cylon::Status::OK();
}

cylon::Status ResolveLocIndices(const std::vector<void *> &input_indices,
                                const std::shared_ptr<cylon::BaseIndex> &index,
                                std::vector<int64_t> &output_indices) {
  cylon::Status status;
  for (size_t ix = 0; ix < input_indices.size(); ix++) {
    std::vector<int64_t> filter_ix;
    void *val = input_indices.at(ix);

    status = index->LocationByValue(val, filter_ix);
    if (!status.is_ok()) {
      LOG(ERROR) << "Error in retrieving indices!";
      return status;
    }
    for (size_t iy = 0; iy < filter_ix.size(); iy++) {
      output_indices.push_back(filter_ix.at(iy));
    }
  }
  return cylon::Status::OK();
}

cylon::Status GetTableFromIndices(const std::shared_ptr<cylon::Table> &input_table,
                                  const std::vector<int64_t> &filter_indices,
                                  std::shared_ptr<cylon::Table> &output) {

  std::shared_ptr<arrow::Array> out_idx;
  arrow::Status arrow_status;
  auto ctx = input_table->GetContext();
  auto pool = cylon::ToArrowPool(ctx);
  arrow::compute::ExecContext fn_ctx(pool);
  arrow::Int64Builder idx_builder(pool);
  const arrow::Datum input_table_datum(input_table->get_table());

  arrow_status = idx_builder.AppendValues(filter_indices);

  if (!arrow_status.ok()) {
    LOG(ERROR) << "Error occurred in appending filter indices";
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(arrow_status);
  }

  arrow_status = idx_builder.Finish(&out_idx);

  if (!arrow_status.ok()) {
    LOG(ERROR) << "Error occurred in creating Arrow filter indices";
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(arrow_status);
  }

  const arrow::Datum filter_indices_datum(out_idx);

  arrow::Result<arrow::Datum> result = arrow::compute::Take(input_table_datum,
                                                            filter_indices_datum,
                                                            arrow::compute::TakeOptions::Defaults(),
                                                            &fn_ctx);

  std::shared_ptr<arrow::Table> filter_table;
  filter_table = result.ValueOrDie().table();
  if (!result.status().ok()) {
    LOG(ERROR) << "Error occurred in subset retrieval from table";
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
  }

  cylon::Table::FromArrowTable(ctx, filter_table, output);

  return cylon::Status::OK();
}

cylon::Status GetTableByLocIndex(const void *indices,
                                 const std::shared_ptr<cylon::Table> &input_table,
                                 std::shared_ptr<cylon::BaseIndex> &index,
                                 std::shared_ptr<cylon::Table> &output,
                                 std::vector<int64_t> &filter_indices) {
  //LOG(INFO) << "GetTableByLocIndex";
  cylon::Status status_build;
  auto ctx = input_table->GetContext();
  auto input_artb = input_table->get_table();
  std::shared_ptr<arrow::Table> out_arrow;
  status_build = index->LocationByValue(indices, input_artb, filter_indices, out_arrow);

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error occurred in retrieving indices!";
    return status_build;
  }

  status_build = cylon::Table::FromArrowTable(ctx, out_arrow, output);

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error creating Table";
    return status_build;
  }

  return cylon::Status::OK();
}

cylon::Status cylon::LocIndexer::loc(const void *start_index,
                                     const void *end_index,
                                     const int column_index,
                                     const std::shared_ptr<Table> &input_table,
                                     std::shared_ptr<cylon::Table> &output) {

  Status status_build;
  auto index = input_table->GetIndex();
  std::shared_ptr<cylon::Table> temp_output;
  int64_t s_index = -1;
  int64_t e_index = -1;

  status_build = GetLocFilterIndices(start_index, end_index, index, s_index, e_index);

  std::cout << ">>>>>" << s_index << ", " << e_index << ", index_size: " << index->GetIndexArray()->length() << std::endl;

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error occurred in filtering indices from table";
    return status_build;
  }
  std::vector<int> filter_columns = {column_index};

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

  status_build = SetIndexForLocResultTable(index, s_index, e_index, output, indexing_schema_);

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error occurred in setting index for output table";
    return status_build;
  }

  return cylon::Status::OK();
}
cylon::Status cylon::LocIndexer::loc(const void *start_index,
                                     const void *end_index,
                                     const int start_column_index,
                                     const int end_column_index,
                                     const std::shared_ptr<Table> &input_table,
                                     std::shared_ptr<cylon::Table> &output) {

  Status status_build;
  auto index = input_table->GetIndex();
  std::shared_ptr<cylon::Table> temp_output;
  int64_t s_index, e_index = -1;

  status_build = GetLocFilterIndices(start_index, end_index, index, s_index, e_index);

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

  status_build = SetIndexForLocResultTable(index, s_index, e_index, output, indexing_schema_);

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error occurred in setting index for output table";
    return status_build;
  }

  return cylon::Status::OK();
}
cylon::Status cylon::LocIndexer::loc(const void *start_index,
                                     const void *end_index,
                                     const std::vector<int> &columns,
                                     const std::shared_ptr<Table> &input_table,
                                     std::shared_ptr<cylon::Table> &output) {

  Status status_build;
  std::shared_ptr<cylon::Table> temp_output;
  int64_t s_index, e_index = -1;
  auto index = input_table->GetIndex();
  status_build = GetLocFilterIndices(start_index, end_index, index, s_index, e_index);

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

  status_build = SetIndexForLocResultTable(index, s_index, e_index, output, indexing_schema_);

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error occurred in setting index for output table";
    return status_build;
  }

  return cylon::Status::OK();
}
cylon::Status cylon::LocIndexer::loc(const void *indices,
                                     const int column_index,
                                     const std::shared_ptr<Table> &input_table,
                                     std::shared_ptr<cylon::Table> &output) {

  Status status_build;
  std::shared_ptr<cylon::Table> temp_output;
  auto index = input_table->GetIndex();
  std::vector<int64_t> filter_indices;
  status_build = GetTableByLocIndex(indices, input_table, index, temp_output, filter_indices);

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error occurred in creating table by index!";
    return status_build;
  }

  std::vector<int> filter_columns = {column_index};

  status_build = FilterColumnsFromTable(temp_output, filter_columns, output);

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error occurred in creating table!";
    return status_build;
  }

  status_build = SetIndexForLocResultTable(index, filter_indices, output, indexing_schema_);

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error occurred in setting index for output table";
    return status_build;
  }

  return cylon::Status::OK();
}
cylon::Status cylon::LocIndexer::loc(const std::vector<void *> &indices,
                                     const int start_column_index,
                                     const int end_column_index,
                                     const std::shared_ptr<Table> &input_table,
                                     std::shared_ptr<cylon::Table> &output) {
  Status status;
  std::vector<int64_t> filter_indices;
  std::shared_ptr<cylon::Table> temp_table;
  auto index = input_table->GetIndex();
  status = ResolveLocIndices(indices, index, filter_indices);

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

  status = GetColumnIndicesFromLimits(start_column_index, end_column_index, filter_columns);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in getting column indices from boundaries";
    return status;
  }

  status = FilterColumnsFromTable(temp_table, filter_columns, output);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in creating table from selected columns";
    return status;
  }

  status = SetIndexForLocResultTable(index, filter_indices, output, indexing_schema_);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in setting index for output table";
    return status;
  }

  return cylon::Status::OK();
}
cylon::Status cylon::LocIndexer::loc(const std::vector<void *> &indices,
                                     const std::vector<int> &columns,
                                     const std::shared_ptr<Table> &input_table,
                                     std::shared_ptr<cylon::Table> &output) {

  Status status;
  std::vector<int64_t> filter_indices;
  std::shared_ptr<cylon::Table> temp_table;
  auto index = input_table->GetIndex();
  status = ResolveLocIndices(indices, index, filter_indices);

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

  status = SetIndexForLocResultTable(index, filter_indices, output, indexing_schema_);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in setting index for output table";
    return status;
  }

  return cylon::Status::OK();
}

cylon::Status cylon::LocIndexer::loc(const void *indices,
                                     const std::vector<int> &columns,
                                     const std::shared_ptr<Table> &input_table,
                                     std::shared_ptr<cylon::Table> &output) {
  Status status_build;
  std::shared_ptr<cylon::Table> temp_output;
  auto index = input_table->GetIndex();
  std::vector<int64_t> filter_indices;
  status_build = GetTableByLocIndex(indices, input_table, index, temp_output, filter_indices);

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error occurred in creating table by index!";
    return status_build;
  }

  status_build = FilterColumnsFromTable(temp_output, columns, output);

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error occurred in filtering columns from Table!";
    return status_build;
  }

  status_build = SetIndexForLocResultTable(index, filter_indices, output, indexing_schema_);

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error occurred in setting index for output table";
    return status_build;
  }

  return cylon::Status::OK();
}
cylon::Status cylon::LocIndexer::loc(const void *indices,
                                     const int start_column,
                                     const int end_column,
                                     const std::shared_ptr<Table> &input_table,
                                     std::shared_ptr<cylon::Table> &output) {

  Status status_build;
  std::shared_ptr<cylon::Table> temp_output;
  auto index = input_table->GetIndex();
  std::vector<int64_t> filter_indices;
  status_build = GetTableByLocIndex(indices, input_table, index, temp_output, filter_indices);

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error occurred in creating table by index!";
    return status_build;
  }

  std::vector<int> columns;

  status_build = GetColumnIndicesFromLimits(start_column, end_column, columns);

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error occurred in creating filter columns from boundaries!";
    return status_build;
  }

  status_build = FilterColumnsFromTable(temp_output, columns, output);

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error occurred in filtering columns from Table!";
    return status_build;
  }

  status_build = SetIndexForLocResultTable(index, filter_indices, output, indexing_schema_);

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error occurred in setting index for output table";
    return status_build;
  }

  return cylon::Status::OK();

}
cylon::Status cylon::LocIndexer::loc(const std::vector<void *> &indices,
                                     const int column,
                                     const std::shared_ptr<Table> &input_table,
                                     std::shared_ptr<cylon::Table> &output) {
  Status status;
  std::vector<int64_t> filter_indices;
  std::shared_ptr<cylon::Table> temp_table;
  auto index = input_table->GetIndex();
  status = ResolveLocIndices(indices, index, filter_indices);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in resolving indices for table filtering";
    return status;
  }

  status = GetTableFromIndices(input_table, filter_indices, temp_table);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in creating table from filter indices";
    return status;
  }

  std::vector<int> columns = {column};

  status = FilterColumnsFromTable(temp_table, columns, output);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in creating table from selected columns";
    return status;
  }

  status = SetIndexForLocResultTable(index, filter_indices, output, indexing_schema_);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in setting index for output table";
    return status;
  }

  return cylon::Status::OK();
}

cylon::IndexingSchema cylon::LocIndexer::GetIndexingSchema() {
  return indexing_schema_;
}

/**
 * ILocIndexer implementations
 * */

//static cylon::Status CreateRangeIndexForIloc(std::shared_ptr<cylon::BaseIndex> &index,
//                                             std::shared_ptr<cylon::Table> &table) {
//  cylon::Status status;
//  if (std::shared_ptr<cylon::RangeIndex> r = std::dynamic_pointer_cast<cylon::RangeIndex>(index)) {
//    index = r;
//  } else {
//    status = cylon::IndexUtil::BuildRangeIndex(index, table);
//    if (!status.is_ok()) {
//      LOG(ERROR) << "Error occurred in adding Range index in iloc operation";
//      return status;
//    }
//  }
//  return cylon::Status::OK();
//}

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


cylon::Status cylon::ILocIndexer::loc(const void *start_index,
                                      const void *end_index,
                                      const int column_index,
                                      const std::shared_ptr<Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  //LOG(INFO) << "ILOC Mode 1";
  cylon::Status status;
  std::shared_ptr<cylon::BaseIndex> range_index;
  std::shared_ptr<cylon::Table> temp_out;
  int64_t start_index_pos = *((int64_t *) start_index);
  int64_t end_index_pos = *((int64_t *) end_index);

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
  status = cylon::IndexUtil::BuildRangeIndex(output, range_index);

  output->Set_Index(range_index, false);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in creating range index for output table";
    return status;
  }

  return Status::OK();
}
cylon::Status cylon::ILocIndexer::loc(const void *start_index,
                                      const void *end_index,
                                      const int start_column_index,
                                      const int end_column_index,
                                      const std::shared_ptr<Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {

  //LOG(INFO) << "ILOC Mode 2";
  cylon::Status status;
  std::shared_ptr<cylon::BaseIndex> range_index;
  std::shared_ptr<cylon::Table> temp_out;
  int64_t start_index_pos = *((int64_t *) start_index);
  int64_t end_index_pos = *((int64_t *) end_index);

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
  status = cylon::IndexUtil::BuildRangeIndex(output, range_index);

  output->Set_Index(range_index, false);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in creating range index for output table";
    return status;
  }

  return Status::OK();
}
cylon::Status cylon::ILocIndexer::loc(const void *start_index,
                                      const void *end_index,
                                      const std::vector<int> &columns,
                                      const std::shared_ptr<Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  //LOG(INFO) << "ILOC Mode 3";
  cylon::Status status;
  std::shared_ptr<cylon::BaseIndex> range_index;
  std::shared_ptr<cylon::Table> temp_out;
  int64_t start_index_pos = *((int64_t *) start_index);
  int64_t end_index_pos = *((int64_t *) end_index);

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
  status = cylon::IndexUtil::BuildRangeIndex(output, range_index);

  output->Set_Index(range_index, false);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in creating range index for output table";
    return status;
  }

  return Status::OK();
}
cylon::Status cylon::ILocIndexer::loc(const void *indices,
                                      const int column_index,
                                      const std::shared_ptr<Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  //LOG(INFO) << "ILOC Mode 4";
  cylon::Status status;
  std::shared_ptr<cylon::BaseIndex> range_index;
  std::shared_ptr<cylon::Table> temp_out;

  int64_t index_pos = *((int64_t *) indices);

  std::vector<int64_t> filter_indices;
  filter_indices.push_back(index_pos);

  status = GetTableFromIndices(input_table, filter_indices, temp_out);

  if (!status.is_ok()) {
    LOG(ERROR) << "Filtering table from indices failed!";
    return status;
  }

  std::vector<int> columns = {column_index};

  status = FilterColumnsFromTable(temp_out, columns, output);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in filtering table based on columns";
    return status;
  }
  // default of ILocIndex based operation is a range index
  status = cylon::IndexUtil::BuildRangeIndex(output, range_index);

  output->Set_Index(range_index, false);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in creating range index for output table";
    return status;
  }

  return Status::OK();

}
cylon::Status cylon::ILocIndexer::loc(const void *indices,
                                      const int start_column,
                                      const int end_column,
                                      const std::shared_ptr<Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  //LOG(INFO) << "ILOC Mode 5";
  cylon::Status status;
  std::shared_ptr<cylon::BaseIndex> range_index;
  std::shared_ptr<cylon::Table> temp_out;

  int64_t index_pos = *((int64_t *) indices);

  std::vector<int64_t> filter_indices;
  filter_indices.push_back(index_pos);

  status = GetTableFromIndices(input_table, filter_indices, temp_out);

  if (!status.is_ok()) {
    LOG(ERROR) << "Filtering table from indices failed!";
    return status;
  }

  // filter columns include both boundaries
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
  status = cylon::IndexUtil::BuildRangeIndex(output, range_index);

  output->Set_Index(range_index, false);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in creating range index for output table";
    return status;
  }

  return Status::OK();
}
cylon::Status cylon::ILocIndexer::loc(const void *indices,
                                      const std::vector<int> &columns,
                                      const std::shared_ptr<Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  //LOG(INFO) << "ILOC Mode 6";
  cylon::Status status;
  std::shared_ptr<cylon::BaseIndex> range_index;
  std::shared_ptr<cylon::Table> temp_out;

  int64_t index_pos = *((int64_t *) indices);

  std::vector<int64_t> filter_indices;
  filter_indices.push_back(index_pos);

  status = GetTableFromIndices(input_table, filter_indices, temp_out);

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
  status = cylon::IndexUtil::BuildRangeIndex(output, range_index);

  output->Set_Index(range_index, false);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in creating range index for output table";
    return status;
  }

  return Status::OK();
}
cylon::Status cylon::ILocIndexer::loc(const std::vector<void *> &indices,
                                      const int column,
                                      const std::shared_ptr<Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  //LOG(INFO) << "ILoc Mode 7";
  Status status;
  std::shared_ptr<cylon::BaseIndex> range_index;
  std::vector<int64_t> i_indices;
  std::shared_ptr<cylon::Table> temp_out;
  status = ResolveILocIndices(indices, i_indices);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in iloc index resolving";
  }

  status = GetTableFromIndices(input_table, i_indices, temp_out);

  if (!status.is_ok()) {
    LOG(ERROR) << "Filtering table from indices failed!";
    return status;
  }

  std::vector<int> columns{column};

  status = FilterColumnsFromTable(temp_out, columns, output);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in filtering table based on index range";
    return status;
  }
  // default of ILocIndex based operation is a range index
  status = cylon::IndexUtil::BuildRangeIndex(output, range_index);

  output->Set_Index(range_index, false);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in creating range index for output table";
    return status;
  }

  return Status::OK();
}
cylon::Status cylon::ILocIndexer::loc(const std::vector<void *> &indices,
                                      const int start_column_index,
                                      const int end_column_index,
                                      const std::shared_ptr<Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  //LOG(INFO) << "ILOC Mode 8";
  Status status;
  std::shared_ptr<cylon::BaseIndex> range_index;
  std::vector<int64_t> i_indices;
  std::shared_ptr<cylon::Table> temp_out;
  status = ResolveILocIndices(indices, i_indices);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in iloc index resolving";
  }

  status = GetTableFromIndices(input_table, i_indices, temp_out);

  if (!status.is_ok()) {
    LOG(ERROR) << "Filtering table from indices failed!";
    return status;
  }

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
  status = cylon::IndexUtil::BuildRangeIndex(output, range_index);

  output->Set_Index(range_index, false);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in creating range index for output table";
    return status;
  }

  return Status::OK();
}
cylon::Status cylon::ILocIndexer::loc(const std::vector<void *> &indices,
                                      const std::vector<int> &columns,
                                      const std::shared_ptr<Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  //LOG(INFO) << "ILOC Mode 9";
  Status status;
  std::shared_ptr<cylon::BaseIndex> range_index;
  std::vector<int64_t> i_indices;
  std::shared_ptr<cylon::Table> temp_out;
  status = ResolveILocIndices(indices, i_indices);

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
  status = cylon::IndexUtil::BuildRangeIndex(output, range_index);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in creating range index for output table";
    return status;
  }

  output->Set_Index(range_index, false);

  return Status::OK();
}
cylon::IndexingSchema cylon::ILocIndexer::GetIndexingSchema() {
  return indexing_schema_;
}

cylon::Status cylon::CheckIsIndexValueUnique(const void *index_value,
                                             const std::shared_ptr<BaseIndex> &index,
                                             bool &is_unique) {
  auto index_arr = index->GetIndexArray();
  switch (index_arr->type()->id()) {
    case arrow::Type::NA:break;
    case arrow::Type::BOOL:break;
    case arrow::Type::UINT8: return IsIndexValueUnique<arrow::UInt8Type>(index_value, index, is_unique);
    case arrow::Type::INT8: return IsIndexValueUnique<arrow::Int8Type>(index_value, index, is_unique);
    case arrow::Type::UINT16:return IsIndexValueUnique<arrow::UInt16Type>(index_value, index, is_unique);
    case arrow::Type::INT16:return IsIndexValueUnique<arrow::Int16Type>(index_value, index, is_unique);
    case arrow::Type::UINT32:return IsIndexValueUnique<arrow::UInt32Type>(index_value, index, is_unique);
    case arrow::Type::INT32:return IsIndexValueUnique<arrow::Int32Type>(index_value, index, is_unique);
    case arrow::Type::UINT64:return IsIndexValueUnique<arrow::UInt64Type>(index_value, index, is_unique);
    case arrow::Type::INT64:return IsIndexValueUnique<arrow::Int64Type>(index_value, index, is_unique);
    case arrow::Type::HALF_FLOAT:return IsIndexValueUnique<arrow::HalfFloatType>(index_value, index, is_unique);
    case arrow::Type::FLOAT:return IsIndexValueUnique<arrow::FloatType>(index_value, index, is_unique);
    case arrow::Type::DOUBLE:return IsIndexValueUnique<arrow::DoubleType>(index_value, index, is_unique);
    case arrow::Type::STRING:
      return IsIndexValueUnique<arrow::StringType, arrow::util::string_view>(index_value,
                                                                             index,
                                                                             is_unique);
    case arrow::Type::BINARY:break;
    case arrow::Type::FIXED_SIZE_BINARY:break;
    case arrow::Type::DATE32:return IsIndexValueUnique<arrow::Date32Type>(index_value, index, is_unique);
    case arrow::Type::DATE64:return IsIndexValueUnique<arrow::Date64Type>(index_value, index, is_unique);
    case arrow::Type::TIMESTAMP:return IsIndexValueUnique<arrow::TimestampType>(index_value, index, is_unique);
    case arrow::Type::TIME32:return IsIndexValueUnique<arrow::Time32Type>(index_value, index, is_unique);
    case arrow::Type::TIME64:return IsIndexValueUnique<arrow::Time64Type>(index_value, index, is_unique);
    case arrow::Type::INTERVAL_MONTHS:break;
    case arrow::Type::INTERVAL_DAY_TIME:break;
    case arrow::Type::DECIMAL:break;
    case arrow::Type::LIST:break;
    case arrow::Type::STRUCT:break;
    case arrow::Type::SPARSE_UNION:break;
    case arrow::Type::DENSE_UNION:break;
    case arrow::Type::DICTIONARY:break;
    case arrow::Type::MAP:break;
    case arrow::Type::EXTENSION:break;
    case arrow::Type::FIXED_SIZE_LIST:break;
    case arrow::Type::DURATION:break;
    case arrow::Type::LARGE_STRING:break;
    case arrow::Type::LARGE_BINARY:break;
    case arrow::Type::LARGE_LIST:break;
    case arrow::Type::MAX_ID:break;
  }
  return cylon::Status(cylon::Code::Invalid, "Invalid arrow data type");
}

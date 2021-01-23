
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
  if (std::shared_ptr<cylon::RangeIndex> rx = std::dynamic_pointer_cast<cylon::RangeIndex>(index)) {
    LOG(INFO) << "Set Index for location output with RangeIndex";
    loc_index = std::make_shared<cylon::RangeIndex>(0, sub_index_locations.size(), 1, pool);
  } else {
    LOG(INFO) << "Set Index for location output with Non-RangeIndex";
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

  if (std::shared_ptr<cylon::RangeIndex> rx = std::dynamic_pointer_cast<cylon::RangeIndex>(index)) {
    LOG(INFO) << "Set Index for location output with RangeIndex";
    loc_index = std::make_shared<cylon::RangeIndex>(0, end_pos - start_pos, 1, pool);
  } else {
    LOG(INFO) << "Set Index for location output with Non-RangeIndex";
    auto index_arr = index->GetIndexArray();
    sub_index_arr = index_arr->Slice(start_pos, (end_pos - start_pos + 1));
    BuildIndexFromArrayByKernel(indexing_schema, sub_index_arr, pool, loc_index);
  }

  output->Set_Index(loc_index, false);

  return cylon::Status::OK();
}

cylon::Status GetLocFilterIndices(void *start_index,
                                  void *end_index,
                                  int64_t &s_index,
                                  int64_t &e_index,
                                  std::shared_ptr<cylon::BaseIndex> &index) {
  cylon::Status status1, status2, status_build;
  std::shared_ptr<arrow::Table> out_artb;

  status1 = index->LocationByValue(start_index, s_index);
  status2 = index->LocationByValue(end_index, e_index);

  if (!(status1.is_ok() and status2.is_ok())) {
    LOG(ERROR) << "Error occurred in extracting indices!";
    return cylon::Status(cylon::Code::IndexError);
  }

  return cylon::Status::OK();
}

cylon::Status SliceTableByRange(int64_t start_index,
                                int64_t end_index,
                                std::shared_ptr<cylon::Table> &input_table,
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

cylon::Status GetColumnIndicesFromLimits(int &start_column, int &end_column, std::vector<int> &selected_columns) {

  if (start_column > end_column) {
    LOG(ERROR) << "Invalid column boundaries";
    return cylon::Status(cylon::Code::Invalid);
  }

  for (int s = start_column; s <= end_column; s++) {
    selected_columns.push_back(s);
  }

  return cylon::Status::OK();
}

cylon::Status FilterColumnsFromTable(std::shared_ptr<cylon::Table> &input_table,
                                     std::vector<int> &filter_columns,
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

cylon::Status ResolveLocIndices(std::vector<void *> &input_indices,
                                std::shared_ptr<cylon::BaseIndex> &index,
                                std::vector<int64_t> &output_indices) {
  cylon::Status status;
  for (size_t ix = 0; ix < input_indices.size(); ix++) {
    std::vector<int64_t> filter_ix;
    void *val = input_indices.at(ix);

    status = index->LocationByValue(&val, filter_ix);
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

cylon::Status GetTableFromIndices(std::shared_ptr<cylon::Table> &input_table,
                                  std::vector<int64_t> &filter_indices,
                                  std::shared_ptr<cylon::Table> &output) {

  std::shared_ptr<arrow::Array> out_idx;
  arrow::Status arrow_status;
  auto ctx = input_table->GetContext();
  auto pool = cylon::ToArrowPool(ctx);
  arrow::compute::ExecContext fn_ctx(pool);
  arrow::Int64Builder idx_builder(pool);
  const arrow::Datum input_table_datum(input_table->get_table());

  idx_builder.AppendValues(filter_indices);
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

cylon::Status GetTableByLocIndex(void *indices,
                                 std::shared_ptr<cylon::Table> &input_table,
                                 std::shared_ptr<cylon::BaseIndex> &index,
                                 std::shared_ptr<cylon::Table> &output,
                                 std::vector<int64_t> &filter_indices) {
  LOG(INFO) << "GetTableByLocIndex";
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

cylon::Status cylon::LocIndexer::loc(void *start_index,
                                     void *end_index,
                                     int column_index,
                                     std::shared_ptr<cylon::Table> &input_table,
                                     std::shared_ptr<cylon::Table> &output) {

  Status status_build;
  auto index = input_table->GetIndex();
  std::shared_ptr<cylon::Table> temp_output;
  int64_t s_index, e_index = -1;

  status_build = GetLocFilterIndices(start_index, end_index, s_index, e_index, index);

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
cylon::Status cylon::LocIndexer::loc(void *start_index,
                                     void *end_index,
                                     int start_column_index,
                                     int end_column_index,
                                     std::shared_ptr<cylon::Table> &input_table,
                                     std::shared_ptr<cylon::Table> &output) {

  Status status_build;
  auto index = input_table->GetIndex();
  std::shared_ptr<cylon::Table> temp_output;
  int64_t s_index, e_index = -1;

  status_build = GetLocFilterIndices(start_index, end_index, s_index, e_index, index);

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
cylon::Status cylon::LocIndexer::loc(void *start_index,
                                     void *end_index,
                                     std::vector<int> &columns,
                                     std::shared_ptr<cylon::Table> &input_table,
                                     std::shared_ptr<cylon::Table> &output) {

  Status status_build;
  std::shared_ptr<cylon::Table> temp_output;
  int64_t s_index, e_index = -1;
  auto index = input_table->GetIndex();
  status_build = GetLocFilterIndices(start_index, end_index, s_index, e_index, index);

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
cylon::Status cylon::LocIndexer::loc(void *indices,
                                     int column_index,
                                     std::shared_ptr<cylon::Table> &input_table,
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
cylon::Status cylon::LocIndexer::loc(std::vector<void *> &indices,
                                     int start_column_index,
                                     int end_column_index,
                                     std::shared_ptr<cylon::Table> &input_table,
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
cylon::Status cylon::LocIndexer::loc(std::vector<void *> &indices,
                                     std::vector<int> &columns,
                                     std::shared_ptr<cylon::Table> &input_table,
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

cylon::Status cylon::LocIndexer::loc(void *indices,
                                     std::vector<int> &columns,
                                     std::shared_ptr<cylon::Table> &input_table,
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
cylon::Status cylon::LocIndexer::loc(void *indices,
                                     int start_column,
                                     int end_column,
                                     std::shared_ptr<cylon::Table> &input_table,
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
cylon::Status cylon::LocIndexer::loc(std::vector<void *> &indices,
                                     int column,
                                     std::shared_ptr<cylon::Table> &input_table,
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



cylon::Status cylon::ILocIndexer::loc(void *start_index,
                                      void *end_index,
                                      int column_index,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {

//  auto index = input_table->GetIndex();
//
//  int64_t s_index = reinterpret_cast<int64_t>(start_index);
//  int64_t e_index = reinterpret_cast<int64_t>(end_index);

  return Status::OK();
}
cylon::Status cylon::ILocIndexer::loc(void *start_index,
                                      void *end_index,
                                      int start_column_index,
                                      int end_column_index,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  return Status();
}
cylon::Status cylon::ILocIndexer::loc(void *start_index,
                                      void *end_index,
                                      std::vector<int> &columns,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  return Status();
}
cylon::Status cylon::ILocIndexer::loc(void *indices,
                                      int column_index,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  return Status();
}
cylon::Status cylon::ILocIndexer::loc(void *indices,
                                      int start_column,
                                      int end_column,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  return Status();
}
cylon::Status cylon::ILocIndexer::loc(void *indices,
                                      std::vector<int> &columns,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  return Status();
}
cylon::Status cylon::ILocIndexer::loc(std::vector<void *> &indices,
                                      int column,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  return Status();
}
cylon::Status cylon::ILocIndexer::loc(std::vector<void *> &indices,
                                      int start_column_index,
                                      int end_column_index,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  return Status();
}
cylon::Status cylon::ILocIndexer::loc(std::vector<void *> &indices,
                                      std::vector<int> &columns,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  return Status();
}
cylon::IndexingSchema cylon::ILocIndexer::GetIndexingSchema() {
  return indexing_schema_;
}




#include "indexer.hpp"

cylon::Status GetFilterIndices(void *start_index,
                               void *end_index,
                               int64_t &s_index,
                               int64_t &e_index,
                               std::shared_ptr<cylon::BaseIndex> &index) {
  cylon::Status status1, status2, status_build;
  std::shared_ptr<arrow::Table> out_artb;

  status1 = index->Find(start_index, s_index);
  status2 = index->Find(end_index, e_index);

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

cylon::Status ResolveIndices(std::vector<void *> &input_indices,
                             std::shared_ptr<cylon::BaseIndex> &index,
                             std::vector<int64_t> &output_indices) {
  cylon::Status status;
  for (size_t ix = 0; ix < input_indices.size(); ix++) {
    std::vector<int64_t> filter_ix;
    void *val = input_indices.at(ix);

    status = index->Find(&val, filter_ix);
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

cylon::Status GetTableByIndex(void *indices,
                              std::shared_ptr<cylon::Table> &input_table,
                              std::shared_ptr<cylon::BaseIndex> &index,
                              std::shared_ptr<cylon::Table> &output) {

  cylon::Status status_build;
  auto ctx = input_table->GetContext();
  auto input_artb = input_table->get_table();
  std::shared_ptr<arrow::Table> out_arrow;
  status_build = index->Find(indices, input_artb, out_arrow);

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

cylon::Status cylon::BaseIndexer::loc(void *start_index,
                                      void *end_index,
                                      int column_index,
                                      std::shared_ptr<BaseIndex> &index,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {

  Status status_build;
  std::shared_ptr<cylon::Table> temp_output;
  int64_t s_index, e_index = -1;

  status_build = GetFilterIndices(start_index, end_index, s_index, e_index, index);

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

  return cylon::Status::OK();
}
cylon::Status cylon::BaseIndexer::loc(void *start_index,
                                      void *end_index,
                                      int start_column_index,
                                      int end_column_index,
                                      std::shared_ptr<BaseIndex> &index,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {

  Status status_build;
  std::shared_ptr<cylon::Table> temp_output;
  int64_t s_index, e_index = -1;

  status_build = GetFilterIndices(start_index, end_index, s_index, e_index, index);

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

  return cylon::Status::OK();
}
cylon::Status cylon::BaseIndexer::loc(void *start_index,
                                      void *end_index,
                                      std::vector<int> &columns,
                                      std::shared_ptr<BaseIndex> &index,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {

  Status status_build;
  std::shared_ptr<cylon::Table> temp_output;
  int64_t s_index, e_index = -1;

  status_build = GetFilterIndices(start_index, end_index, s_index, e_index, index);

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

  return cylon::Status::OK();
}
cylon::Status cylon::BaseIndexer::loc(void *indices,
                                      int column_index,
                                      std::shared_ptr<BaseIndex> &index,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {

  Status status_build;
  std::shared_ptr<cylon::Table> temp_output;

  status_build = GetTableByIndex(indices, input_table, index, temp_output);

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

  return cylon::Status::OK();
}
cylon::Status cylon::BaseIndexer::loc(std::vector<void *> &indices,
                                      int start_column_index,
                                      int end_column_index,
                                      std::shared_ptr<BaseIndex> &index,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  Status status;
  std::vector<int64_t> filter_indices;
  std::shared_ptr<cylon::Table> temp_table;

  status = ResolveIndices(indices, index, filter_indices);

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

  return cylon::Status::OK();
}
cylon::Status cylon::BaseIndexer::loc(std::vector<void *> &indices,
                                      std::vector<int> &columns,
                                      std::shared_ptr<BaseIndex> &index,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {

  Status status;
  std::vector<int64_t> filter_indices;
  std::shared_ptr<cylon::Table> temp_table;

  status = ResolveIndices(indices, index, filter_indices);

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

  return cylon::Status::OK();
}

cylon::Status cylon::BaseIndexer::loc(void *indices,
                                      std::vector<int> &columns,
                                      std::shared_ptr<BaseIndex> &index,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  Status status_build;
  std::shared_ptr<cylon::Table> temp_output;

  status_build = GetTableByIndex(indices, input_table, index, temp_output);

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error occurred in creating table by index!";
    return status_build;
  }

  status_build = FilterColumnsFromTable(temp_output, columns, output);

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error occurred in filtering columns from Table!";
    return status_build;
  }

  return cylon::Status::OK();
}

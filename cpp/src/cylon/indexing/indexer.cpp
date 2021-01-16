
#include "indexer.hpp"

cylon::Status GetFilterIndices(void *start_index,
                               void *end_index,
                               int64_t &s_index,
                               int64_t &e_index,
                               std::shared_ptr<cylon::BaseIndex> &index) {
  cylon::Status status1, status2, status_build;
  std::shared_ptr<arrow::Table> out_artb, out_artb_res;

  status1 = index->Find(start_index, s_index);
  status2 = index->Find(end_index, e_index);

  if (!(status1.is_ok() and status2.is_ok())) {
    LOG(ERROR) << "Error occurred in extracting indices!";
    return cylon::Status(cylon::Code::IndexError);
  }
  // to include last index to the result
  e_index++;
  return cylon::Status::OK();
}

cylon::Status SliceTableByRange(int64_t start_index,
                                int64_t end_index,
                                std::vector<int> &filter_columns,
                                std::shared_ptr<cylon::Table> &input_table,
                                std::shared_ptr<cylon::Table> &output) {

  cylon::Status status_build;
  std::shared_ptr<arrow::Table> out_artb, out_artb_res;

  auto artb = input_table->get_table();
  auto ctx = input_table->GetContext();

  out_artb = artb->Slice(start_index, end_index);

  arrow::Result<std::shared_ptr<arrow::Table>> result = out_artb->SelectColumns(filter_columns);

  if (!result.status().ok()) {
    LOG(ERROR) << "Column selection failed in loc operation!";
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
  }

  status_build = cylon::Table::FromArrowTable(ctx, result.ValueOrDie(), output);

  if (!status_build.is_ok()) {
    LOG(ERROR) << "Error occurred in creating loc output table";
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
  int64_t s_index, e_index = -1;

  status_build = GetFilterIndices(start_index, end_index, s_index, e_index, index);

  if (!status_build.is_ok()) {
    return status_build;
  }
  std::vector<int> filter_columns = {column_index};

  status_build = SliceTableByRange(s_index, e_index, filter_columns, input_table, output);

  if (!status_build.is_ok()) {
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
  int64_t s_index, e_index = -1;

  status_build = GetFilterIndices(start_index, end_index, s_index, e_index, index);

  if (!status_build.is_ok()) {
    return status_build;
  }

  // filter columns include both boundaries
  std::vector<int> filter_columns(end_column_index - start_column_index + 1);

  for (int s = s_index; s <= e_index; s++) {
    filter_columns.push_back(s);
  }

  status_build = SliceTableByRange(s_index, e_index, filter_columns, input_table, output);

  if (!status_build.is_ok()) {
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
  int64_t s_index, e_index = -1;

  status_build = GetFilterIndices(start_index, end_index, s_index, e_index, index);

  if (!status_build.is_ok()) {
    return status_build;
  }

  status_build = SliceTableByRange(s_index, e_index, columns, input_table, output);

  if (!status_build.is_ok()) {
    return status_build;
  }

  return cylon::Status::OK();
}
cylon::Status cylon::BaseIndexer::loc(void *indices,
                                      int column_index,
                                      std::shared_ptr<BaseIndex> &index,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {


  return cylon::Status::OK();
}
cylon::Status cylon::BaseIndexer::loc(std::vector<void *> &indices,
                                      int start_column_index,
                                      int end_column_index,
                                      std::shared_ptr<BaseIndex> &index,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  return cylon::Status();
}
cylon::Status cylon::BaseIndexer::loc(std::vector<void *> &indices,
                                      std::vector<int> &columns,
                                      std::shared_ptr<BaseIndex> &index,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  return cylon::Status();
}

cylon::Status cylon::BaseIndexer::loc(void *indices,
                                      std::vector<int> &columns,
                                      std::shared_ptr<BaseIndex> &index,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  return cylon::Status();
}

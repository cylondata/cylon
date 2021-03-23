//
// Created by vibhatha on 3/18/21.
//

#include "arrow_indexer.hpp"
#include "index_utils.hpp"

cylon::Status cylon::ArrowLocIndexer::loc(const std::shared_ptr<arrow::Array> &start_index,
										  const std::shared_ptr<arrow::Array> &end_index,
										  const int column_index,
										  const std::shared_ptr<Table> &input_table,
										  std::shared_ptr<cylon::Table> &output) {

  auto start_scalar = start_index->GetScalar(0).ValueOrDie();
  auto end_scalar = end_index->GetScalar(0).ValueOrDie();

  auto index_array_ = input_table->GetIndex()->GetIndexArray();
  int64_t start_idx = -1;
  int64_t end_idx = -1;


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
cylon::Status cylon::ArrowLocIndexer::loc(const std::shared_ptr<arrow::Array> &start_index,
										  const std::shared_ptr<arrow::Array> &end_index,
										  const int start_column_index,
										  const int end_column_index,
										  const std::shared_ptr<Table> &input_table,
										  std::shared_ptr<cylon::Table> &output) {
  return Status();
}
cylon::Status cylon::ArrowLocIndexer::loc(const std::shared_ptr<arrow::Array> &start_index,
										  const std::shared_ptr<arrow::Array> &end_index,
										  const std::vector<int> &columns,
										  const std::shared_ptr<Table> &input_table,
										  std::shared_ptr<cylon::Table> &output) {
  return Status();
}
cylon::Status cylon::ArrowLocIndexer::loc(const std::shared_ptr<arrow::Array> &indices,
										  const int column_index,
										  const std::shared_ptr<Table> &input_table,
										  std::shared_ptr<cylon::Table> &output) {
  return Status();
}
cylon::Status cylon::ArrowLocIndexer::loc(const std::shared_ptr<arrow::Array> &indices,
										  const int start_column,
										  const int end_column,
										  const std::shared_ptr<Table> &input_table,
										  std::shared_ptr<cylon::Table> &output) {
  return Status();
}
cylon::Status cylon::ArrowLocIndexer::loc(const std::shared_ptr<arrow::Scalar> &indices,
										  const std::vector<int> &columns,
										  const std::shared_ptr<Table> &input_table,
										  std::shared_ptr<cylon::Table> &output) {
  return Status();
}
cylon::IndexingSchema cylon::ArrowLocIndexer::GetIndexingSchema() {
  return Linear;
}

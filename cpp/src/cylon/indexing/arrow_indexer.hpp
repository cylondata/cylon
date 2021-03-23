//
// Created by vibhatha on 3/18/21.
//

#ifndef CYLON_SRC_CYLON_INDEXING_ARROW_INDEXER_H_
#define CYLON_SRC_CYLON_INDEXING_ARROW_INDEXER_H_

#include "index.hpp"
#include "table.hpp"



namespace cylon {

class ArrowBaseIndexer {

 public:
  explicit ArrowBaseIndexer() {

  }

  virtual Status loc(const std::shared_ptr<arrow::Array> &start_index,
					 const std::shared_ptr<arrow::Array> &end_index,
					 const int column_index,
					 const std::shared_ptr<Table> &input_table,
					 std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(const std::shared_ptr<arrow::Array> &start_index,
					 const std::shared_ptr<arrow::Array> &end_index,
					 const int start_column_index,
					 const int end_column_index,
					 const std::shared_ptr<Table> &input_table,
					 std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(const std::shared_ptr<arrow::Array> &start_index,
					 const std::shared_ptr<arrow::Array> &end_index,
					 const std::vector<int> &columns,
					 const std::shared_ptr<Table> &input_table,
					 std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(const std::shared_ptr<arrow::Array> &indices,
					 const int column_index,
					 const std::shared_ptr<Table> &input_table,
					 std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(const std::shared_ptr<arrow::Array> &indices,
					 const int start_column,
					 const int end_column,
					 const std::shared_ptr<Table> &input_table,
					 std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(const std::shared_ptr<arrow::Scalar> &indices,
					 const std::vector<int> &columns,
					 const std::shared_ptr<Table> &input_table,
					 std::shared_ptr<cylon::Table> &output) = 0;

  virtual IndexingSchema GetIndexingSchema() = 0;

};

class ArrowLocIndexer : public ArrowBaseIndexer {

 public:
  ArrowLocIndexer(IndexingSchema indexing_schema) : ArrowBaseIndexer(), indexing_schema_(indexing_schema) {

  };

  Status loc(const std::shared_ptr<arrow::Array> &start_index,
			 const std::shared_ptr<arrow::Array> &end_index,
			 const int column_index,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  Status loc(const std::shared_ptr<arrow::Array> &start_index,
			 const std::shared_ptr<arrow::Array> &end_index,
			 const int start_column_index,
			 const int end_column_index,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  Status loc(const std::shared_ptr<arrow::Array> &start_index,
			 const std::shared_ptr<arrow::Array> &end_index,
			 const std::vector<int> &columns,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  Status loc(const std::shared_ptr<arrow::Array> &indices,
			 const int column_index,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  Status loc(const std::shared_ptr<arrow::Array> &indices,
			 const int start_column,
			 const int end_column,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  Status loc(const std::shared_ptr<arrow::Scalar> &indices,
			 const std::vector<int> &columns,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  IndexingSchema GetIndexingSchema() override;

 private:
  IndexingSchema indexing_schema_;

};

}

#endif //CYLON_SRC_CYLON_INDEXING_ARROW_INDEXER_H_

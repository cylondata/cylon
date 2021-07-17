
#ifndef CYLON_SRC_CYLON_INDEXING_INDEXER_H_
#define CYLON_SRC_CYLON_INDEXING_INDEXER_H_

#include <cylon/indexing/index.hpp>
#include <cylon/table.hpp>

namespace cylon {

bool CheckIsIndexValueUnique(const std::shared_ptr<arrow::Scalar> &index_value,
							 const std::shared_ptr<BaseArrowIndex> &index);

/**
 * Loc operations
 * */

class ArrowBaseIndexer {

 public:
  explicit ArrowBaseIndexer() {

  }

  virtual Status loc(const std::shared_ptr<arrow::Scalar> &start_index,
					 const std::shared_ptr<arrow::Scalar> &end_index,
					 const int column_index,
					 const std::shared_ptr<Table> &input_table,
					 std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(const std::shared_ptr<arrow::Scalar> &start_index,
					 const std::shared_ptr<arrow::Scalar> &end_index,
					 const int start_column_index,
					 const int end_column_index,
					 const std::shared_ptr<Table> &input_table,
					 std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(const std::shared_ptr<arrow::Scalar> &start_index,
					 const std::shared_ptr<arrow::Scalar> &end_index,
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

  virtual Status loc(const std::shared_ptr<arrow::Array> &indices,
					 const std::vector<int> &columns,
					 const std::shared_ptr<Table> &input_table,
					 std::shared_ptr<cylon::Table> &output) = 0;

  virtual IndexingType GetIndexingType() = 0;

};

class ArrowLocIndexer : public ArrowBaseIndexer {

 public:
  explicit ArrowLocIndexer(IndexingType indexing_type) : ArrowBaseIndexer(), indexing_type_(indexing_type) {

  };

  Status loc(const std::shared_ptr<arrow::Scalar> &start_index,
			 const std::shared_ptr<arrow::Scalar> &end_index,
			 const int column_index,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  Status loc(const std::shared_ptr<arrow::Scalar> &start_index,
			 const std::shared_ptr<arrow::Scalar> &end_index,
			 const int start_column_index,
			 const int end_column_index,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  Status loc(const std::shared_ptr<arrow::Scalar> &start_index,
			 const std::shared_ptr<arrow::Scalar> &end_index,
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
  Status loc(const std::shared_ptr<arrow::Array> &indices,
			 const std::vector<int> &columns,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  IndexingType GetIndexingType() override;

 private:
  IndexingType indexing_type_;

};

/**
 * iLoc operations
 * */

class ArrowILocIndexer : public ArrowLocIndexer {
 public:
  ArrowILocIndexer(IndexingType indexing_type);

  Status loc(const std::shared_ptr<arrow::Scalar> &start_index,
			 const std::shared_ptr<arrow::Scalar> &end_index,
			 const int column_index,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  Status loc(const std::shared_ptr<arrow::Scalar> &start_index,
			 const std::shared_ptr<arrow::Scalar> &end_index,
			 const int start_column_index,
			 const int end_column_index,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  Status loc(const std::shared_ptr<arrow::Scalar> &start_index,
			 const std::shared_ptr<arrow::Scalar> &end_index,
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
  Status loc(const std::shared_ptr<arrow::Array> &indices,
			 const std::vector<int> &columns,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  IndexingType GetIndexingType() override;

 private:
  IndexingType indexing_type_;
};

}

#endif //CYLON_SRC_CYLON_INDEXING_INDEXER_H_

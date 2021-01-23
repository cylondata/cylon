
#ifndef CYLON_SRC_CYLON_INDEXING_INDEXER_H_
#define CYLON_SRC_CYLON_INDEXING_INDEXER_H_

#include "index.hpp"
#include "table.hpp"

namespace cylon {

template<typename Base, typename T>
inline bool instanceof(const T*) {
  return std::is_base_of<Base, T>::value;
}

enum IndexingSchema {
  Range = 0,
  Linear = 1,
  Hash = 2,
  BinaryTree = 3,
  BTree = 4,
};

class BaseIndexer {

 public:
  explicit BaseIndexer()  {

  }

  virtual Status loc(void *start_index,
                     void *end_index,
                     int column_index,
                     std::shared_ptr<cylon::Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(void *start_index,
                     void *end_index,
                     int start_column_index,
                     int end_column_index,
                     std::shared_ptr<cylon::Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(void *start_index,
                     void *end_index,
                     std::vector<int> &columns,
                     std::shared_ptr<cylon::Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(void *indices,
                     int column_index,
                     std::shared_ptr<cylon::Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(void *indices,
                     int start_column,
                     int end_column,
                     std::shared_ptr<cylon::Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(void *indices,
                     std::vector<int> &columns,
                     std::shared_ptr<cylon::Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(std::vector<void *> &indices,
                     int column,
                     std::shared_ptr<cylon::Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(std::vector<void *> &indices,
                     int start_column_index,
                     int end_column_index,
                     std::shared_ptr<cylon::Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(std::vector<void *> &indices,
                     std::vector<int> &columns,
                     std::shared_ptr<cylon::Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual IndexingSchema GetIndexingSchema() = 0;

};

/**
 * Loc operations
 * */


class LocIndexer : public BaseIndexer {

 public:
  LocIndexer(IndexingSchema indexing_schema) : BaseIndexer(), indexing_schema_(indexing_schema) {

  };

  Status loc(void *start_index,
             void *end_index,
             int column_index,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(void *start_index,
             void *end_index,
             int start_column_index,
             int end_column_index,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(void *start_index,
             void *end_index,
             std::vector<int> &columns,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(void *indices,
             int column_index,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(void *indices,
             int start_column,
             int end_column,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(void *indices,
             std::vector<int> &columns,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(std::vector<void *> &indices,
             int column,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(std::vector<void *> &indices,
             int start_column_index,
             int end_column_index,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(std::vector<void *> &indices,
             std::vector<int> &columns,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;

  IndexingSchema GetIndexingSchema() override;

 private:
  IndexingSchema indexing_schema_;

};

/**
 * iLoc operations
 * */

class ILocIndexer : public BaseIndexer {
 public:
  ILocIndexer(IndexingSchema indexing_schema) : BaseIndexer(), indexing_schema_(indexing_schema) {

  };

  Status loc(void *start_index,
             void *end_index,
             int column_index,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(void *start_index,
             void *end_index,
             int start_column_index,
             int end_column_index,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(void *start_index,
             void *end_index,
             std::vector<int> &columns,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(void *indices,
             int column_index,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(void *indices,
             int start_column,
             int end_column,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(void *indices,
             std::vector<int> &columns,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(std::vector<void *> &indices,
             int column,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(std::vector<void *> &indices,
             int start_column_index,
             int end_column_index,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(std::vector<void *> &indices,
             std::vector<int> &columns,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;

  IndexingSchema GetIndexingSchema() override;

 private:
  IndexingSchema indexing_schema_;
};

}

#endif //CYLON_SRC_CYLON_INDEXING_INDEXER_H_

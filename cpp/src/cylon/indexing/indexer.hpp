
#ifndef CYLON_SRC_CYLON_INDEXING_INDEXER_H_
#define CYLON_SRC_CYLON_INDEXING_INDEXER_H_

#include "index.hpp"
#include "table.hpp"

namespace cylon {

class BaseIndexer {

 public:
  explicit BaseIndexer() {

  }

  /**
   * Loc operations
   * */

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

  /**
   * iLoc operations
   * */

};

class LocHashIndexer : public BaseIndexer {

 public:
  LocHashIndexer() : BaseIndexer() {

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

};

class ILocHashIndexer : public BaseIndexer{
 public:
  ILocHashIndexer() : BaseIndexer() {

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
};

}

#endif //CYLON_SRC_CYLON_INDEXING_INDEXER_H_

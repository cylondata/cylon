
#ifndef CYLON_SRC_CYLON_INDEXING_INDEXER_H_
#define CYLON_SRC_CYLON_INDEXING_INDEXER_H_

#include "index.hpp"
#include "table.hpp"

namespace cylon {

class BaseIndexer {

 public:

  static Status loc(void *start_index,
                    void *end_index,
                    int column_index,
                    std::shared_ptr<BaseIndex> &index,
                    std::shared_ptr<cylon::Table> &input_table,
                    std::shared_ptr<cylon::Table> &output);

  static Status loc(void *start_index,
                    void *end_index,
                    int start_column_index,
                    int end_column_index,
                    std::shared_ptr<BaseIndex> &index,
                    std::shared_ptr<cylon::Table> &input_table,
                    std::shared_ptr<cylon::Table> &output);

  static Status loc(void *start_index,
                    void *end_index,
                    std::vector<int> &columns,
                    std::shared_ptr<BaseIndex> &index,
                    std::shared_ptr<cylon::Table> &input_table,
                    std::shared_ptr<cylon::Table> &output);

  static Status loc(void *indices,
                    int column_index,
                    std::shared_ptr<BaseIndex> &index,
                    std::shared_ptr<cylon::Table> &input_table,
                    std::shared_ptr<cylon::Table> &output);

  static Status loc(void *indices,
                    std::vector<int> &columns,
                    std::shared_ptr<BaseIndex> &index,
                    std::shared_ptr<cylon::Table> &input_table,
                    std::shared_ptr<cylon::Table> &output);

  static Status loc(std::vector<void *> &indices,
                    int start_column_index,
                    int end_column_index,
                    std::shared_ptr<BaseIndex> &index,
                    std::shared_ptr<cylon::Table> &input_table,
                    std::shared_ptr<cylon::Table> &output);

  static Status loc(std::vector<void *> &indices,
                    std::vector<int> &columns,
                    std::shared_ptr<BaseIndex> &index,
                    std::shared_ptr<cylon::Table> &input_table,
                    std::shared_ptr<cylon::Table> &output);

};
}

#endif //CYLON_SRC_CYLON_INDEXING_INDEXER_H_

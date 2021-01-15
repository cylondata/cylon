
#ifndef CYLON_SRC_CYLON_INDEXING_INDEXER_H_
#define CYLON_SRC_CYLON_INDEXING_INDEXER_H_

#include "index.hpp"
#include "table.hpp"

namespace cylon {

class BaseIndexer {

 public:
  BaseIndexer(std::shared_ptr<cylon::BaseIndex> index);

  void loc(void *start_index,
           void *end_index,
           int column_index,
           std::shared_ptr<cylon::Table> &input_table,
           std::shared_ptr<cylon::Table> &output);

  void loc(void *start_index,
           void *end_index,
           int start_column_index,
           int end_column_index,
           std::shared_ptr<cylon::Table> &input_table,
           std::shared_ptr<cylon::Table> &output);

  void loc(void *start_index,
           void *end_index,
           std::vector<int> &columns,
           std::shared_ptr<cylon::Table> &input_table,
           std::shared_ptr<cylon::Table> &output);

 private:
  std::shared_ptr<BaseIndexer> base_index_;


};

template<class ARROW_T, typename CTYPE = typename ARROW_T::c_type>
class Indexer{

};

}

#endif //CYLON_SRC_CYLON_INDEXING_INDEXER_H_

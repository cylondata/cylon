

#ifndef CYLON_SRC_CYLON_INDEXING_BUILDER_H_
#define CYLON_SRC_CYLON_INDEXING_BUILDER_H_

#include "index.hpp"
#include "status.hpp"
#include "table.hpp"

namespace cylon {

class IndexUtil {

 public:
  static Status Build(std::shared_ptr<cylon::BaseIndex> &index,
                      std::shared_ptr<cylon::Table> &input,
                      int index_column,
                      bool drop_index,
                      std::shared_ptr<cylon::Table> &output);

  static Status Find(std::shared_ptr<cylon::BaseIndex> &index,
                     std::shared_ptr<cylon::Table> &find_table,
                     void *value,
                     int index_column,
                     std::shared_ptr<cylon::Table> &out);

};

}

#endif //CYLON_SRC_CYLON_INDEXING_BUILDER_H_


#include "indexer.hpp"

cylon::Status cylon::BaseIndexer::loc(void *start_index,
                                      void *end_index,
                                      int column_index,
                                      std::shared_ptr<BaseIndex> &index,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {



  return cylon::Status::OK();
}
cylon::Status cylon::BaseIndexer::loc(void *start_index,
                                      void *end_index,
                                      int start_column_index,
                                      int end_column_index,
                                      std::shared_ptr<BaseIndex> &index,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  return cylon::Status();
}
cylon::Status cylon::BaseIndexer::loc(void *start_index,
                                      void *end_index,
                                      std::vector<int> &columns,
                                      std::shared_ptr<BaseIndex> &index,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  return cylon::Status();
}
cylon::Status cylon::BaseIndexer::loc(std::vector<void *> &indices,
                                      int column_index,
                                      std::shared_ptr<BaseIndex> &index,
                                      std::shared_ptr<cylon::Table> &input_table,
                                      std::shared_ptr<cylon::Table> &output) {
  return cylon::Status();
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

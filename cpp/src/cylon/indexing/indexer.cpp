
#include "indexer.hpp"

cylon::BaseIndexer::BaseIndexer(std::shared_ptr<cylon::BaseIndex> index) {

}
void cylon::BaseIndexer::loc(void *start_index,
                             void *end_index,
                             int column_index,
                             std::shared_ptr<cylon::Table> &input_table,
                             std::shared_ptr<cylon::Table> &output) {

}
void cylon::BaseIndexer::loc(void *start_index,
                             void *end_index,
                             int start_column_index,
                             int end_column_index,
                             std::shared_ptr<cylon::Table> &input_table,
                             std::shared_ptr<cylon::Table> &output) {

}
void cylon::BaseIndexer::loc(void *start_index,
                             void *end_index,
                             std::vector<int> &columns,
                             std::shared_ptr<cylon::Table> &input_table,
                             std::shared_ptr<cylon::Table> &output) {

}

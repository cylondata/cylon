
#include <unordered_set>
#include "index.hpp"

namespace cylon {

Index::Index(const std::unordered_multiset<int64_t, TableRowIndexHash, TableRowIndexComparator> &index_set, const std::vector<int> &col_ids) : index_set_(
    index_set), col_ids_(0) {
  for(auto col_id: col_ids){
    col_ids_.push_back(col_id);
  }
  std::cout << "Size of MultiSet " << index_set_.size() << std::endl;
}
const std::unordered_multiset<int64_t, TableRowIndexHash, TableRowIndexComparator> &Index::GetIndexSet() const {
  return index_set_;
}
const std::vector<int32_t> &Index::GetColIds() const {
  return col_ids_;
}
}
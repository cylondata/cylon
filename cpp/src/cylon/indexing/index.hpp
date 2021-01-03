
#ifndef CYLON_SRC_CYLON_INDEXING_INDEX_H_
#define CYLON_SRC_CYLON_INDEXING_INDEX_H_

#include <arrow/arrow_comparator.hpp>

namespace cylon{

class Index {
 public:
  Index(const std::unordered_multiset<int64_t, TableRowIndexHash, TableRowIndexComparator> &index_set, const std::vector<int> &col_ids);

 private:
  std::unordered_multiset<int64_t, TableRowIndexHash, TableRowIndexComparator> index_set_;
  std::vector<int> col_ids_;
 public:
  const std::unordered_multiset<int64_t, TableRowIndexHash, TableRowIndexComparator> &GetIndexSet() const;
  const std::vector<int> &GetColIds() const;
};

}



#endif //CYLON_SRC_CYLON_INDEXING_INDEX_H_

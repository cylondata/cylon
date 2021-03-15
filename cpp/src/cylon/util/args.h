#ifndef C4CE5DE6_2779_4FEB_B0EB_E159C043F124
#define C4CE5DE6_2779_4FEB_B0EB_E159C043F124

#include <arrow/api.h>

#include <memory>

union ColumnArg {
  std::shared_ptr<arrow::Table> table;
  int64_t col_idx;
  std::vector<int64_t> col_idxs;
};

#endif /* C4CE5DE6_2779_4FEB_B0EB_E159C043F124 */

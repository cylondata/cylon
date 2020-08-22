
#ifndef CYLON_SRC_CYLON_OPS_KERNELS_UNION_HPP_
#define CYLON_SRC_CYLON_OPS_KERNELS_UNION_HPP_
#include <vector>
#include <table.hpp>
#include <ops/kernels/utils/RowComparator.hpp>

namespace cylon {
namespace kernel {
class Union {
 private:
  std::vector<std::shared_ptr<arrow::Table >> tables{};
  std::vector<std::shared_ptr<std::vector<int64_t>>> indices_from_tabs{};
  std::shared_ptr<arrow::Schema> schema;
  shared_ptr<CylonContext> ctx;

  std::unordered_set<std::pair<int8_t, int64_t>, RowComparator, RowComparator> *rows_set;

 public:
  ~Union();
  Union(std::shared_ptr<cylon::CylonContext> ctx,
        std::shared_ptr<arrow::Schema> schema,
        int64_t expected_rows);
  void InsertTable(std::shared_ptr<cylon::Table> table);
  cylon::Status Finalize(std::shared_ptr<cylon::Table> &result);
};
}
}
#endif //CYLON_SRC_CYLON_OPS_KERNELS_UNION_HPP_

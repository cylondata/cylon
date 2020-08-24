#ifndef CYLON_SRC_CYLON_OPS_UNION_OP_HPP_
#define CYLON_SRC_CYLON_OPS_UNION_OP_HPP_
#include <ops/kernels/utils/RowComparator.hpp>
#include <ops/kernels/union.hpp>
#include "parallel_op.hpp"

namespace cylon {
class UnionOpConfig {
 private:
  int64_t expected_rows = 100000;

 public:
  int64_t GetExpectedRows() const;
  void SetExpectedRows(int64_t expected_rows);
};

class UnionOp : public Op {

 private:
  std::shared_ptr<UnionOpConfig> config;
  cylon::kernel::Union *union_kernel;

 public:
  UnionOp(std::shared_ptr<cylon::CylonContext> ctx,
          std::shared_ptr<arrow::Schema> schema,
          int id, std::function<int(int)> router,
          std::shared_ptr<ResultsCallback> callback,
          std::shared_ptr<UnionOpConfig> config);
  bool Execute(int tag, std::shared_ptr<Table> table) override;

  void OnParentsFinalized() override;
  bool Finalize() override;
};

}
#endif //CYLON_SRC_CYLON_OPS_UNION_OP_HPP_

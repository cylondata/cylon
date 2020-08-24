#ifndef CYLON_SRC_CYLON_OPS_DIS_UNION_OP_H_
#define CYLON_SRC_CYLON_OPS_DIS_UNION_OP_H_

#include "parallel_op.hpp"

namespace cylon {

class DisUnionOpConfig {

};

class DisUnionOp : public Op {

 private:
  std::shared_ptr<DisUnionOpConfig> config;

 public:
  DisUnionOp(std::shared_ptr<cylon::CylonContext> ctx, std::shared_ptr<arrow::Schema> schema,
             int id,
             std::shared_ptr<ResultsCallback> callback,
             std::shared_ptr<DisUnionOpConfig>
             config);
  bool Execute(int tag, std::shared_ptr<Table> table) override;

  void OnParentsFinalized() override;
  bool Finalize() override;
};
}
#endif //CYLON_SRC_CYLON_OPS_DIS_UNION_OP_H_

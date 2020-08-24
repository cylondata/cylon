#ifndef CYLON_SRC_CYLON_OPS_KERNELS_SHUFFLE_H_
#define CYLON_SRC_CYLON_OPS_KERNELS_SHUFFLE_H_

#include "parallel_op.hpp"
namespace cylon {

class ShuffleOpConfig {

};

class ShuffleOp : public Op {

 private:
  cylon::ArrowCallback *arrow_callback_;
  cylon::ArrowAllToAll *all_to_all_;
  bool communication_done = false;

 public:
  ShuffleOp(std::shared_ptr<cylon::CylonContext> ctx,
            std::shared_ptr<arrow::Schema> schema,
            int id, std::function<int(int)> router,
            std::shared_ptr<ResultsCallback> callback,
            std::shared_ptr<ShuffleOpConfig> config);

  void Progress();

  bool Execute(int tag, std::shared_ptr<Table> table) override;

  bool Finalize() override;

  void OnParentsFinalized() override ;
};
}

#endif //CYLON_SRC_CYLON_OPS_KERNELS_SHUFFLE_H_

#ifndef CYLON_SRC_CYLON_OPS_PARTITION_OP_H_
#define CYLON_SRC_CYLON_OPS_PARTITION_OP_H_
#include "parallel_op.h"

namespace cylon {

class PartitionOp : public Op {
 private:
  static const int PARTITION_OP = 2;
  int no_of_partitions = 1;
  std::vector<int> hash_columns;
  std::shared_ptr<CylonContext> ctx_;

 public:
  static const int INPUT_Q = 0;

  PartitionOp(int id, std::function<int(int)> router, std::shared_ptr<ResultsCallback> callback);
  PartitionOp(std::function<int(int)> router, std::shared_ptr<ResultsCallback> callback);
  void init(std::shared_ptr<CylonContext> ctx, std::shared_ptr<OpConfig> op_config);
  void execute(int tag, std::shared_ptr<Table> table);
  bool ready();
};
}

#endif //CYLON_SRC_CYLON_OPS_PARTITION_OP_H_

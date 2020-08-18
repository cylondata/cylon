#ifndef CYLON_SRC_CYLON_OPS_PARTITION_OP_H_
#define CYLON_SRC_CYLON_OPS_PARTITION_OP_H_
#include "parallel_op.h"

namespace cylon {

class PartitionOpConfig {
 private:
  int no_of_partitions;
  std::shared_ptr<std::vector<int>> hash_columns;

 public:
  PartitionOpConfig(int no_of_partitions, std::shared_ptr<std::vector<int>> hash_columns);
  int NoOfPartitions();
  std::shared_ptr<std::vector<int>> HashColumns();
};

class PartitionOp : public Op {
 private:
  static const int PARTITION_OP = 2;

  std::shared_ptr<PartitionOpConfig> config;

 public:
  static const int INPUT_Q = 0;

  PartitionOp(std::shared_ptr<cylon::CylonContext> ctx,
              int id, std::function<int(int)> router,
              std::shared_ptr<ResultsCallback> callback,
              std::shared_ptr<PartitionOpConfig> config);
  void execute(int tag, std::shared_ptr<Table> table) override;
};
}

#endif //CYLON_SRC_CYLON_OPS_PARTITION_OP_H_

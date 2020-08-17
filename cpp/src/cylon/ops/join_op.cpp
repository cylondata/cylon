#include "join_op.h"
#include "partition_op.h"

cylon::JoinOp::JoinOp(std::function<int(int)> router,
                      std::shared_ptr<ResultsCallback> callback) : Op(JOIN_OP, router, callback) {
  auto left_partition = new PartitionOp(LEFT, [](int tag) {
    return 0;
  }, callback);

  left_partition->IsComplete();
  //auto right_partition = new PartitionOp(RIGHT, callback);

//  this->AddChild(left_partition);
//  this->AddChild(right_partition);
//
//  auto local_join = new LocalJoin(LOCAL_JOIN);
//
//  // left and right goes in two directions until the local join
//  for (auto rel:{LEFT, RIGHT}) {
//    auto partition = this->GetChild(rel);
//
//    auto left_shuffle = new ShuffleOp(SHUFFLE);
//    partition->AddChild(left_shuffle);
//
//    auto sort = new SortOp(SORT);
//    left_shuffle->AddChild(sort);
//
//    sort->AddChild(local_join);
//  }
}
void cylon::JoinOp::execute(int tag, std::shared_ptr<Table> table) {
  // pass the left tables
  this->DrainQueueToChild(LEFT, LEFT, 0);
  this->DrainQueueToChild(RIGHT, RIGHT, 1);
}

bool cylon::JoinOp::Ready() {
  return true;
}
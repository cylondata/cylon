#ifndef CYLON_SRC_CYLON_OPS_PARALLEL_OP_H_
#define CYLON_SRC_CYLON_OPS_PARALLEL_OP_H_

#include <memory>
#include <table.hpp>

/**
 * Assumptions
 * 1. Queue, Map lookups will never fail
 */
namespace cylon {
class Op {
 protected:
  int id;
  std::unordered_map<int, std::queue<std::shared_ptr<cylon::Table>> *> queues{};
  int inputs_count = 0;
  std::unordered_map<int, cylon::Op *> children{};

  Op *GetChild(int tag);

  void DrainQueueToChild(int queue, int child, int tag);

  std::queue<std::shared_ptr<cylon::Table>> *GetQueue(int tag);

 public:
  Op(int id);

  void insert(int tag, std::shared_ptr<cylon::Table> table);

  /**
   * This is the logic of this op
   */
  virtual void execute();

  virtual bool ready();

  void init(cylon::CylonContext *ctx);

  void progress();

  bool isComplete();

  ~Op();

  cylon::Op *AddChild(cylon::Op *child);
};

class LocalJoin : public Op {
 public:
  LocalJoin(int id) : Op(id) {

  }

  // insert, 2 - LEFT, 3 - RIGHT
};

class SortOp : public Op {
 public:
  SortOp(int id) : Op(id) {

  }

  void init(CylonContext *ctx) {
    // initialize merge sort
  }
};

class ShuffleOp : public Op {
 public:
  ShuffleOp(int id) : Op(id) {

  }

  void init(CylonContext *ctx) {
    // initialize all to all
    // post whatever we get
  }
};
// ND arr
class PartitionOp : public Op {
 public:
  PartitionOp(int id) : Op(id) {

  }

  void init(CylonContext *ctx) {
    // initialize hash partitions
    // split the table to chunks
    // do the hash partitioning on chunks and pass the results to the next step
  }
};

class JoinOp : public Op {
 private:
  static const int JOIN_OP = 0;
  static const int LEFT = 2;
  static const int RIGHT = 3;

  static const int SHUFFLE = 4;
  static const int SORT = 5;
  static const int LOCAL_JOIN = 6;

 public:
  JoinOp() : Op(JoinOp::JOIN_OP) {
    auto left_partition = new PartitionOp(LEFT);
    auto right_partition = new PartitionOp(RIGHT);

    this->AddChild(left_partition);
    this->AddChild(right_partition);

    auto local_join = new LocalJoin(LOCAL_JOIN);

    // left and right goes in two directions until the local join
    for (auto rel:{LEFT, RIGHT}) {
      auto partition = this->GetChild(rel);

      auto left_shuffle = new ShuffleOp(SHUFFLE);
      partition->AddChild(left_shuffle);

      auto sort = new SortOp(SORT);
      left_shuffle->AddChild(sort);

      sort->AddChild(local_join);
    }

  }

  void execute() override {
    // pass the left tables
    this->DrainQueueToChild(LEFT, LEFT, 0);
    this->DrainQueueToChild(RIGHT, RIGHT, 1);

  }

  bool ready() override {
    return true;
  }
};
}

#endif //CYLON_SRC_CYLON_OPS_PARALLEL_OP_H_

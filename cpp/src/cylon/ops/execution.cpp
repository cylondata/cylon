#include <list>
#include "execution.hpp"
#include "parallel_op.hpp"

namespace cylon {

/**
 * This execution performs a round robin execution where execution chances will be fairly allocated for all the Ops
 */
class RoundRobinExecution : public Execution {
 private:
  const std::vector<cylon::Op *> &ops;
  std::vector<std::size_t> indices;
  std::size_t current_index;
 public:
  RoundRobinExecution(const std::vector<cylon::Op *> &ops) : ops(ops) {
    for (std::size_t i = 0; i < ops.size(); i++) {
      indices.push_back(i);
    }
  }

  bool IsComplete() override {
    bool completed = ops[indices[current_index]];
    if (completed) {
      indices.erase(indices.begin() + current_index);
    } else {
      current_index++;
    }

    if (current_index == indices.size()) {
      current_index = 0;
    }
    return indices.empty();
  };
};

/**
 * This execution allocate chances based on the priority specified by the user.
 * If Op1 hsa priority 3 and Op2 has priority 1, Op1 will execute 3 times per one execution of Op2
 */
class PriorityExecution : public Execution {
 private:
  RoundRobinExecution *round_robin_execution_;

 public:
  PriorityExecution(const std::vector<std::pair<cylon::Op *, int32_t>> &ops) {
    std::vector<int32_t> counts(ops.size(), 0);
    std::vector<cylon::Op *> ops_(ops.size());
    bool all_added = false;
    while (!all_added) {
      all_added = true;
      for (std::size_t i = 0; i < ops.size(); i++) {
        if (counts[i] < ops[i].second) {
          counts[i]++;
          all_added = false;
        }
      }
    }
    round_robin_execution_ = new RoundRobinExecution(ops_);
  }

  bool IsComplete() override {
    return this->round_robin_execution_->IsComplete();
  }
};

/**
 * This execution performs a BFS over the Op graph
 */
class SequentialExecution : public Execution {

};

}


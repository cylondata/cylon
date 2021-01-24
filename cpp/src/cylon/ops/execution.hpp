#ifndef CYLON_SRC_CYLON_OPS_EXECUTION_HPP_
#define CYLON_SRC_CYLON_OPS_EXECUTION_HPP_
#include <vector>
#include <queue>
#include <cstdint>

namespace cylon {

class Op;

class Execution {
 public:
  virtual bool IsComplete() = 0;
  void WaitForCompletion() {
    while (!this->IsComplete()) {

    }
  }
};

/**
 * This execution performs a round robin execution where execution chances will be fairly allocated for all the Ops
 */
class RoundRobinExecution : public Execution {
 private:
  std::vector<cylon::Op *> ops;
  std::vector<std::size_t> indices;
  std::size_t current_index{};
 public:
  void AddOp(cylon::Op *op);
  bool IsComplete() override;
};

class JoinExecution : public Execution {
private:
  std::vector<cylon::Op *> p_ops;
  std::vector<cylon::Op *> s_ops;
  cylon::Op *join;
  std::vector<std::size_t> p_indices;
  std::vector<std::size_t> s_indices;
  std::size_t current_index{};
  int state = 0;
public:
  void AddP(cylon::Op *op) {
    p_ops.push_back(op);
    this->p_indices.push_back(this->p_ops.size() - 1);
  }
  void AddS(cylon::Op *op) {
    s_ops.push_back(op);
    this->s_indices.push_back(this->s_ops.size() - 1);
  }
  void AddJoin(cylon::Op *op) {
    join = op;
  }
  bool IsComplete() override;
};

/**
 * This execution allocate chances based on the priority specified by the user.
 * If Op1 hsa priority 3 and Op2 has priority 1, Op1 will execute 3 times per one execution of Op2
 */
class PriorityExecution : public Execution {
 private:
  RoundRobinExecution *round_robin_execution_;

 public:
  explicit PriorityExecution();
  void AddOp(cylon::Op *op, int32_t priority);
  bool IsComplete() override;
};

/**
 * This execution performs a BFS over the Op graph
 */
class SequentialExecution : public Execution {
 private:
  std::queue<cylon::Op *> ops;
 public:
  void AddOp(cylon::Op *op);
  bool IsComplete() override;
};
}
#endif //CYLON_SRC_CYLON_OPS_EXECUTION_HPP_

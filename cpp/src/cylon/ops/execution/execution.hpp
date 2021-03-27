#ifndef CYLON_SRC_CYLON_OPS_EXECUTION_HPP_
#define CYLON_SRC_CYLON_OPS_EXECUTION_HPP_
#include <vector>
#include <queue>
#include <cstdint>
#include <cstring>

namespace cylon {

class Op;

// todo keep these in a tree structure rather than adhoc vectors and attach it to cylon::Op::AddChild methods
class Execution {
 public:
  virtual ~Execution() = default;

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
  std::vector<Op *> ops;
  std::vector<size_t> indices;
  size_t current_index{};
 public:
  ~RoundRobinExecution() override;
  void AddOp(Op *op);
  bool IsComplete() override;
};

class JoinExecution : public Execution {
private:
  std::vector<Op *> p_ops;
  std::vector<Op *> s_ops;
  Op *join;
  std::vector<size_t> p_indices;
  std::vector<size_t> s_indices;
  std::size_t current_index{};
  int state = 0;
public:
  ~JoinExecution() override;

  void AddP(Op *op) {
    p_ops.push_back(op);
    p_indices.push_back(p_ops.size() - 1);
  }
  void AddS(Op *op) {
    s_ops.push_back(op);
    s_indices.push_back(s_ops.size() - 1);
  }
  void AddJoin(Op *op) {
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

  ~PriorityExecution() override;

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
  SequentialExecution() : ops(std::queue<cylon::Op *>()) {}

  void AddOp(cylon::Op *op);

  bool IsComplete() override;
};
}
#endif //CYLON_SRC_CYLON_OPS_EXECUTION_HPP_

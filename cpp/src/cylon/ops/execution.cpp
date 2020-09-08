#include "execution.hpp"
#include "parallel_op.hpp"

namespace cylon {

bool RoundRobinExecution::IsComplete() {
  bool completed = ops[indices[current_index]]->IsComplete();
  if (completed) {
    indices.erase(indices.begin() + current_index);
  } else {
    current_index++;
  }

  if (current_index == indices.size()) {
    current_index = 0;
  }
  return indices.empty();
}

void RoundRobinExecution::AddOp(cylon::Op *op) {
  this->ops.push_back(op);
  this->indices.push_back(this->ops.size() - 1);
}

PriorityExecution::PriorityExecution(const std::vector<std::pair<cylon::Op *, int32_t>> &ops) {
  std::vector<int32_t> counts(ops.size(), 0);
  round_robin_execution_ = new RoundRobinExecution();
  bool all_added = false;
  while (!all_added) {
    all_added = true;
    for (std::size_t i = 0; i < ops.size(); i++) {
      if (counts[i] < ops[i].second) {
        counts[i]++;
        round_robin_execution_->AddOp(ops[i].first);
        all_added = false;
      }
    }
  }
}

bool PriorityExecution::IsComplete() {
  return this->round_robin_execution_->IsComplete();
}
}


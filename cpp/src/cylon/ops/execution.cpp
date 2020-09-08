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

PriorityExecution::PriorityExecution() {
  round_robin_execution_ = new RoundRobinExecution();
}

bool PriorityExecution::IsComplete() {
  return this->round_robin_execution_->IsComplete();
}

void PriorityExecution::AddOp(cylon::Op *op, int32_t priority) {
  for (int32_t p = 0; p < priority; p++) {
    this->round_robin_execution_->AddOp(op);
  }
}
void SequentialExecution::AddOp(cylon::Op *op) {
  this->ops.push(op);
}

bool SequentialExecution::IsComplete() {
  cylon::Op *op = this->ops.front();
  if(op->IsComplete()){
    this->ops.pop();
  }
  return this->ops.empty();
}
}


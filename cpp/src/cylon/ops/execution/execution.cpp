#include "execution.hpp"
#include "ops/api/parallel_op.hpp"

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

bool ForkJoinExecution::IsComplete() {
  bool completed;
  switch (state) {
    case 0:completed = p_ops[p_indices[current_index]]->IsComplete();
      if (completed) {
        p_indices.erase(p_indices.begin() + current_index);
      } else {
        current_index++;
      }

      if (current_index == p_indices.size()) {
        current_index = 0;
      }
      if (p_indices.empty()) {
        state = 1;
        current_index = 0;
      }
      break;
    case 1:completed = s_ops[s_indices[current_index]]->IsComplete();
      if (completed) {
        s_indices.erase(s_indices.begin() + current_index);
      } else {
        current_index++;
      }

      if (current_index == s_indices.size()) {
        current_index = 0;
      }
      if (s_indices.empty()) {
        state = 2;
        current_index = 0;
      }
      break;
    case 2:return join->IsComplete();
  }
  return false;
}

ForkJoinExecution::~ForkJoinExecution() {
  // NOTE: p_ops[0] is the head op which holds this execution object.
  // So, don't delete that!
  for (size_t i = 1; i < p_ops.size(); i++) {
    delete p_ops[i];
  }

  for (auto &&o:s_ops) {
    delete o;
  }

  delete join;
}

void RoundRobinExecution::AddOp(cylon::Op *op) {
  this->ops.push_back(op);
  this->indices.push_back(this->ops.size() - 1);
}

RoundRobinExecution::~RoundRobinExecution() {
  // NOTE: p_ops[0] is the head op which holds this execution object.
  // So, don't delete that!
  for (size_t i = 1; i < ops.size(); i++) {
    delete ops[i];
  }
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

PriorityExecution::~PriorityExecution() {
  delete round_robin_execution_;
}

void SequentialExecution::AddOp(cylon::Op *op) {
  this->ops.push(op);
}

bool SequentialExecution::IsComplete() {
  cylon::Op *op = this->ops.front();
  if (op->IsComplete()) {
    this->ops.pop();
  }
  return this->ops.empty();
}

}
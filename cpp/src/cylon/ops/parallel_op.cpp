#include "parallel_op.hpp"

cylon::Op *cylon::Op::GetChild(int tag) {
  return this->children.find(tag)->second;
}

void cylon::Op::DrainQueueToChild(int queue, int child, int tag) {
  auto q = GetQueue(queue);
  auto c = GetChild(child);
  while (!q->empty()) {
    c->InsertTable(tag, q->front());
    q->pop();
  }
}

std::queue<std::shared_ptr<cylon::Table>> *cylon::Op::GetQueue(int tag) {
  return this->queues.find(tag)->second;
}

cylon::Op::Op(std::shared_ptr<cylon::CylonContext> ctx,
              std::shared_ptr<arrow::Schema> schema,
              int id,
              std::function<int(int)> router,
              std::shared_ptr<ResultsCallback> callback) {
  this->ctx_ = ctx;
  this->id = id;
  this->callback = callback;
  this->router = router;
  this->schema = schema;
}

void cylon::Op::InsertTable(int tag, std::shared_ptr<cylon::Table> table) {
  if (queues.find(tag) == queues.end()) {
    queues.insert(std::make_pair<>(tag, new std::queue<std::shared_ptr<cylon::Table>>()));
  }
  this->queues.find(tag)->second->push(table);
  this->inputs_count++;
}

void cylon::Op::Progress() {
  // first process this Op
  if (this->Ready()) {
    for (auto const &q:this->queues) {
      if (!q.second->empty()) {
        bool done = this->Execute(q.first, q.second->front());
        if (done) {
          q.second->pop();
          this->inputs_count--;
          // todo instead of keeping in this queue, partially processed tables can be added to a different queue
          // todo check whether this is the best way to do this. But always assume that the status of the
          // partially processed tables will be kept by the Op implementation
          // std::queue<std::pair<int, std::shared_ptr<cylon::Table>>> partially_processed_queue{};
        }
      }
    }
  }

  // then progress children
  for (auto child: children) {
    child.second->Progress();

    // if no more inputs and finalize has been called, pass finalize to children
    if (this->inputs_count == 0 && this->finalize_inputs_called) {
      child.second->ReportParentCompleted();
    }
  }

  // if this is the final node and if there is nothing left to do post the result

}

bool cylon::Op::IsComplete() {
  for (auto child: children) {
    if (!child.second->IsComplete()) {
      return false;
    }
  }
  // no more inputs will be received && no more items left in the queues
  return this->finalize_inputs_called && this->inputs_count == 0;
}

cylon::Op::~Op() {
  for (auto p: queues) {
    delete p.second;
  }
}

cylon::Op *cylon::Op::AddChild(cylon::Op *child) {
  this->children.insert(std::make_pair(child->id, child));
  child->IncrementParentsCount();
  return this;
}

bool cylon::Op::TerminalCheck(int tag, std::shared_ptr<cylon::Table> table) {
  if (this->children.empty()) {
    this->callback->OnResult(tag, table);
    return true;
  }
  return false;
}

void cylon::Op::InsertToAllChildren(int tag, std::shared_ptr<cylon::Table> table) {
  if (!this->TerminalCheck(tag, table)) {
    for (auto const &child:this->children) {
      child.second->InsertTable(tag, table);
    }
  }
}

void cylon::Op::InsertToChild(int tag, int child, std::shared_ptr<cylon::Table> table) {
  if (!this->TerminalCheck(tag, table)) {
    this->children.find(child)->second->InsertTable(tag, table);
  }
}

bool cylon::Op::Ready() {
  return this->inputs_count > 0;
}

void cylon::Op::IncrementParentsCount() {
  this->parents++;
}

void cylon::Op::ReportParentCompleted() {
  this->finalized_parents++;
  if (this->finalized_parents == 0) {
    this->FinalizeInputs();
  }
}

void cylon::Op::FinalizeInputs() {
  this->finalize_inputs_called = true;
}

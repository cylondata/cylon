#include "parallel_op.h"

typedef std::shared_ptr<cylon::CylonContext> ptr;
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
              int id,
              std::function<int(int)> router,
              std::shared_ptr<ResultsCallback> callback) {
  this->ctx_ = ctx;
  this->id = id;
  this->callback = callback;
  this->router = router;
}

void cylon::Op::InsertTable(int tag, std::shared_ptr<cylon::Table> table) {
  if (queues.find(tag) == queues.end()) {
    queues.insert(std::make_pair<>(tag, new std::queue<std::shared_ptr<cylon::Table>>()));
  }
  this->queues.find(tag)->second->push(table);
  this->inputs_count++;
}

void cylon::Op::Progress() {
  if (this->Ready()) {
    // todo can be changed to incrementally do the processing
    for (auto const &q:this->queues) {
      if (!q.second->empty()) {
        this->execute(q.first, q.second->front());
        q.second->pop();
        this->inputs_count--;
        // todo now yield for another op
      }
    }
  }

  // possible optimization point
  for (auto child: children) {
    child.second->Progress();
  }
}

bool cylon::Op::IsComplete() {
  for (auto child: children) {
    if (!child.second->IsComplete()) {
      return false;
    }
  }
  return true;
}

cylon::Op::~Op() {
  for (auto p: queues) {
    delete p.second;
  }
}

cylon::Op *cylon::Op::AddChild(cylon::Op *child) {
  this->children.insert(std::make_pair(child->id, child));
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
  if (this->children.empty()) {

  }
}

bool cylon::Op::Ready() {
  return this->inputs_count > 0;
}
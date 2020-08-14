#include "parallel_op.h"

cylon::Op *cylon::Op::GetChild(int tag) {
  return this->children.find(tag)->second;
}
void cylon::Op::DrainQueueToChild(int queue, int child, int tag) {
  auto q = GetQueue(queue);
  auto c = GetChild(child);
  while (!q->empty()) {
    c->insert(tag, q->front());
    q->pop();
  }
}

std::queue<std::shared_ptr<cylon::Table>> *cylon::Op::GetQueue(int tag) {
  return this->queues.find(tag)->second;
}

cylon::Op::Op(int id) {
  this->id = id;
}
void cylon::Op::insert(int tag, std::shared_ptr<cylon::Table> table) {
  if (queues.find(tag) == queues.end()) {
    queues.insert(std::make_pair<>(tag, new std::queue<std::shared_ptr<cylon::Table>>()));
  }
  this->queues.find(tag)->second->push(table);
  this->inputs_count++;
}

void cylon::Op::init(cylon::CylonContext *ctx) {
// possible optimization point
  for (auto child: children) {
    child.second->init(ctx);
  }
}

void cylon::Op::progress() {
  if (this->ready()) {
    this->execute();
  }

  // possible optimization point
  for (auto child: children) {
    child.second->progress();
  }
}
bool cylon::Op::isComplete() {
  for (auto child: children) {
    if (!child.second->isComplete()) {
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

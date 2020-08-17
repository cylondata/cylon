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

cylon::Op::Op(int id, std::function<int(int)> router, std::shared_ptr<ResultsCallback> callback) {
  this->id = id;
  this->callback = callback;
  this->router = router;
}

void cylon::Op::insert(int tag, std::shared_ptr<cylon::Table> table) {
  if (queues.find(tag) == queues.end()) {
    queues.insert(std::make_pair<>(tag, new std::queue<std::shared_ptr<cylon::Table>>()));
  }
  this->queues.find(tag)->second->push(table);
  this->inputs_count++;
}

void cylon::Op::init(std::shared_ptr<cylon::CylonContext> ctx, std::shared_ptr<OpConfig> op_config) {
// possible optimization point
  for (auto child: children) {
    child.second->init(ctx, op_config);
  }
}

void cylon::Op::progress() {
  if (this->ready()) {
    // todo can be changed to incrementally do the processing
    for (auto const &q:this->queues) {
      if (!q.second->empty()) {
        this->execute(q.first, q.second->front());
        q.second->pop();

        // todo now yield for another op
      }
    }
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
void cylon::Op::InsertToAllChildren(int tag, std::shared_ptr<cylon::Table> table) {
  for (auto const &child:this->children) {
    child.second->insert(tag, table);
  }
}

cylon::OpConfig *cylon::OpConfig::AddConfig(const string &key, const string &value) {
  this->config.insert(std::make_pair<>(key, value));
  return this;
}

std::string cylon::OpConfig::GetConfig(const string &key, const string &def) {
  return std::string();
}
int64_t cylon::OpConfig::GetLong(const string &key, int64_t defaultValue) {
  auto find = this->config.find(key);
  if (find == this->config.end()) {
    return defaultValue;
  }
  return std::stol(find->second);
}

int32_t cylon::OpConfig::GetDouble(const string &key, double_t defaultValue) {
  auto find = this->config.find(key);
  if (find == this->config.end()) {
    return defaultValue;
  }
  return std::stod(find->second);
}

/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "parallel_op.hpp"

#include <utility>

//cylon::Op *cylon::Op::GetChild(int tag) {
//  return this->children.at(tag);
//}

void cylon::Op::DrainQueueToChild(int queue, int child, int tag) {
  const auto &q = this->queues.at(queue);
  const auto &c = this->children.at(child);
  while (!q->empty()) {
    c->InsertTable(tag, q->front());
    q->pop();
  }
}

//std::queue<std::shared_ptr<cylon::Table>> *cylon::Op::GetQueue(int tag) {
//  return this->queues.at(tag);
//}

cylon::Op::Op(const std::shared_ptr<cylon::CylonContext> &ctx,
              const std::shared_ptr<arrow::Schema> &schema,
              int id,
              ResultsCallback callback,
              bool root_op)
    : callback(std::move(callback)), all_parents_finalized(root_op), id(id), ctx_(ctx), schema(schema) {
}

void cylon::Op::InsertTable(int tag, std::shared_ptr<Table> &table) {
  auto q_iter = queues.find(tag);
  if (q_iter == queues.end()) { // if tag not found -> create queue for tag
    q_iter = queues.emplace(tag, new std::queue<std::shared_ptr<cylon::Table>>()).first;
  }
  q_iter->second->push(table);
  this->inputs_count++;
}

bool cylon::Op::IsComplete() {
  this->did_work = false;

  // first process this Op
  if (!this->finalized && this->inputs_count > 0) {
    for (auto const &q:this->queues) {
      if (!q.second->empty()) {
        bool done = this->Execute(q.first, q.second->front());
        if (done) {
          q.second->pop();
          this->inputs_count--;
          // todo instead of keeping in this queue, partially processed tables
          //  can be added to a different queue
          // todo check whether this is the best way to do this. But always
          //  assume that the status of the
          // partially processed tables will be kept by the Op implementation
          // std::queue<std::pair<int, std::shared_ptr<cylon::Table>>> partially_processed_queue{};
          this->did_work = true;
        }
      }
    }
  }

  // if no more inputs and all parents have finalized, try to finalize this op
  // if finalizing this Op is successful report that to children
  if (!this->finalized && this->inputs_count == 0 && this->all_parents_finalized) {
    // try to finalize this Op
    if (this->Finalize()) {
      for (const auto &child: children) {
        child.second->ReportParentCompleted();
      }
      this->finalized = true;
    }
  }

  return this->finalized;
}

cylon::Op::~Op() {
  // delete queues
  for (auto p: queues) {
    delete p.second;
  }

  // delete children
  for (auto p: children) {
    delete p.second;
  }
}

cylon::Op *cylon::Op::AddChild(cylon::Op *child) {
  children.emplace(child->id, child);
  child->IncrementParentsCount();
  return this;
}

bool cylon::Op::TerminalCheck(int tag, std::shared_ptr<Table> &table) {
  if (children.empty()) {
    callback(tag, table);
    return true;
  }
  return false;
}

void cylon::Op::InsertToAllChildren(int tag, std::shared_ptr<Table> &table) {
  if (!this->TerminalCheck(tag, table)) {
    for (auto const &child:this->children) {
      child.second->InsertTable(tag, table);
    }
  }
}

void cylon::Op::InsertToChild(int child, int tag, std::shared_ptr<Table> &table) {
  if (!this->TerminalCheck(tag, table)) {
    this->children.at(child)->InsertTable(tag, table);
  }
}

void cylon::Op::IncrementParentsCount() {
  this->parents++;
}

void cylon::Op::ReportParentCompleted() {
  this->finalized_parents++;
  if (this->finalized_parents >= parents) {
    this->all_parents_finalized = true;
    this->OnParentsFinalized();
  }
}

bool cylon::Op::DidSomeWork() const {
  return this->did_work;
}

int cylon::Op::GetId() const {
  return id;
}

bool cylon::RootOp::Finalize() {
  return true;
}

void cylon::RootOp::OnParentsFinalized() {
  // do nothing
}

bool cylon::RootOp::Execute(int tag, std::shared_ptr<Table> &table) {
  this->InsertToAllChildren(tag, table);
  return true;
}

void cylon::RootOp::SetExecution(cylon::Execution *execution) {
  execution_ = execution;
}

cylon::Execution *cylon::RootOp::GetExecution() {
  return this->execution_;
}

void cylon::RootOp::WaitForCompletion() {
  execution_->WaitForCompletion();
}

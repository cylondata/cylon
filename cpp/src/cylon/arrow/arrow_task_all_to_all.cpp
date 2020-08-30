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

#include <utility>

#include "./arrow_task_all_to_all.h"

cylon::LogicalTaskPlan::LogicalTaskPlan(std::shared_ptr<vector<int>> task_source,
                                        std::shared_ptr<vector<int>> task_targets,
                                        std::shared_ptr<vector<int>> worker_sources,
                                        std::shared_ptr<vector<int>> worker_targets,
                                        std::shared_ptr<unordered_map<int, int>> task_to_worker)
    : task_source(std::move(task_source)), task_targets(std::move(task_targets)),
      worker_sources(std::move(worker_sources)), worker_targets(std::move(worker_targets)),
      task_to_worker(std::move(task_to_worker)) {

}
const shared_ptr<vector<int>> &cylon::LogicalTaskPlan::GetTaskSource() const {
  return task_source;
}
const shared_ptr<vector<int>> &cylon::LogicalTaskPlan::GetTaskTargets() const {
  return task_targets;
}
const shared_ptr<vector<int>> &cylon::LogicalTaskPlan::GetWorkerSources() const {
  return worker_sources;
}
const shared_ptr<vector<int>> &cylon::LogicalTaskPlan::GetWorkerTargets() const {
  return worker_targets;
}
const shared_ptr<std::unordered_map<int, int>> &cylon::LogicalTaskPlan::GetTaskToWorker() const {
  return task_to_worker;
}

bool cylon::ArrowTaskCallBack::onReceive(int worker_source,
                                         const std::shared_ptr<arrow::Table> &table,
                                         int target_task) {
  return this->onReceive(table, target_task);
}
cylon::ArrowTaskAllToAll::ArrowTaskAllToAll(cylon::CylonContext *ctx,
                                            const cylon::LogicalTaskPlan &plan,
                                            int edgeId,
                                            const shared_ptr<ArrowTaskCallBack> &callback,
                                            const shared_ptr<arrow::Schema> &schema) : ArrowAllToAll(ctx,
                                                                                                     *plan.GetWorkerSources(),
                                                                                                     *plan.GetWorkerTargets(),
                                                                                                     edgeId,
                                                                                                     callback,
                                                                                                     schema),
                                                                                       plan(plan) {

}
int cylon::ArrowTaskAllToAll::InsertTable(shared_ptr<arrow::Table> arrow, int32_t task_target) {
  this->mutex.lock();
  LOG(INFO) << "sending to task  " << task_target << " of worker "<< this->plan.GetTaskToWorker()->find(task_target)->second;
  int inserted = ArrowAllToAll::insert(arrow, this->plan.GetTaskToWorker()->find(task_target)->second, task_target);
  this->mutex.unlock();
  return inserted;
}

bool cylon::ArrowTaskAllToAll::IsComplete() {
  this->mutex.lock();
  auto comp = this->isComplete();
  this->mutex.unlock();
  return comp;
}

void cylon::ArrowTaskAllToAll::WaitForCompletion() {
  while (!this->IsComplete()) {

  }
}

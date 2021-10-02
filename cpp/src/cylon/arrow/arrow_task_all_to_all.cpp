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

#include <cylon/arrow/arrow_task_all_to_all.h>
#include <cylon/util/macros.hpp>

cylon::LogicalTaskPlan::LogicalTaskPlan(std::shared_ptr<std::vector<int>> task_source,
                                        std::shared_ptr<std::vector<int>> task_targets,
                                        std::shared_ptr<std::vector<int>> worker_sources,
                                        std::shared_ptr<std::vector<int>> worker_targets,
                                        std::shared_ptr<std::unordered_map<int, int>> task_to_worker)
    : task_source(std::move(task_source)), task_targets(std::move(task_targets)),
      worker_sources(std::move(worker_sources)), worker_targets(std::move(worker_targets)),
      task_to_worker(std::move(task_to_worker)) {

}
const std::shared_ptr<std::vector<int>> &cylon::LogicalTaskPlan::GetTaskSource() const {
  return task_source;
}
const std::shared_ptr<std::vector<int>> &cylon::LogicalTaskPlan::GetTaskTargets() const {
  return task_targets;
}
const std::shared_ptr<std::vector<int>> &cylon::LogicalTaskPlan::GetWorkerSources() const {
  return worker_sources;
}
const std::shared_ptr<std::vector<int>> &cylon::LogicalTaskPlan::GetWorkerTargets() const {
  return worker_targets;
}
const std::shared_ptr<std::unordered_map<int, int>> &cylon::LogicalTaskPlan::GetTaskToWorker() const {
  return task_to_worker;
}

cylon::ArrowTaskAllToAll::ArrowTaskAllToAll(const std::shared_ptr<CylonContext> &ctx,
                                            const cylon::LogicalTaskPlan &plan,
                                            int edgeId,
                                            ArrowTaskCallBack callback,
                                            const std::shared_ptr<arrow::Schema> &schema)
    : ArrowAllToAll(ctx,
                    *plan.GetWorkerSources(),
                    *plan.GetWorkerTargets(),
                    edgeId,
                    [&callback](int worker_source, const std::shared_ptr<arrow::Table> &table, int target_task) {
                      CYLON_UNUSED(worker_source);
                      return callback(table, target_task);
                    },
                    schema),
      plan(plan) {
}
int cylon::ArrowTaskAllToAll::InsertTable(std::shared_ptr<arrow::Table> &arrow, int32_t task_target) {
  this->mutex.lock();
  LOG(INFO) << "sending to task  " << task_target << " of worker "
            << this->plan.GetTaskToWorker()->find(task_target)->second;
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

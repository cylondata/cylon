#ifndef CYLON_SRC_CYLON_ARROW_ARROW_TASK_ALL_TO_ALL_H_
#define CYLON_SRC_CYLON_ARROW_ARROW_TASK_ALL_TO_ALL_H_

#include <mutex>
#include "arrow_all_to_all.hpp"
#include <glog/logging.h>
namespace cylon {

class LogicalTaskPlan {

 private:
  std::shared_ptr<std::vector<int>> task_source;
  std::shared_ptr<std::vector<int>> task_targets;
  std::shared_ptr<std::vector<int>> worker_sources;
  std::shared_ptr<std::vector<int>> worker_targets;
  std::shared_ptr<std::unordered_map<int, int>> task_to_worker;

 public:
  LogicalTaskPlan(std::shared_ptr<std::vector<int>> task_source,
                  std::shared_ptr<std::vector<int>> task_targets,
                  std::shared_ptr<std::vector<int>> worker_sources,
                  std::shared_ptr<std::vector<int>> worker_targets,
                  std::shared_ptr<std::unordered_map<int,
                                                     int>> task_to_worker);

  const std::shared_ptr<std::vector<int>> &GetTaskSource() const;
  const std::shared_ptr<std::vector<int>> &GetTaskTargets() const;
  const std::shared_ptr<std::vector<int>> &GetWorkerSources() const;
  const std::shared_ptr<std::vector<int>> &GetWorkerTargets() const;
  const std::shared_ptr<std::unordered_map<int, int>> &GetTaskToWorker() const;
};

//class ArrowTaskCallBack : public ArrowCallback {
//  bool onReceive(int worker_source, const std::shared_ptr<arrow::Table> &table, int target_task) override;
//
//  virtual bool onReceive(const std::shared_ptr<arrow::Table> &table, int target) = 0;
//};

using ArrowTaskCallBack = std::function<bool(const std::shared_ptr<arrow::Table> &table, int target)>;

class ArrowTaskAllToAll : public ArrowAllToAll {

 protected:
  std::mutex mutex;
  const LogicalTaskPlan &plan;

 public:
  ArrowTaskAllToAll(std::shared_ptr<cylon::CylonContext> ctx,
                    const LogicalTaskPlan &plan,
                    int edgeId,
                    ArrowTaskCallBack callback,
                    const std::shared_ptr<arrow::Schema> &schema);

  int InsertTable(std::shared_ptr<arrow::Table> &arrow, int32_t task_target);

  bool IsComplete();

  void WaitForCompletion();
};
}

#endif //CYLON_SRC_CYLON_ARROW_ARROW_TASK_ALL_TO_ALL_H_

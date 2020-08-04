#ifndef CYLON_SRC_CYLON_ARROW_ARROW_TASK_ALL_TO_ALL_H_
#define CYLON_SRC_CYLON_ARROW_ARROW_TASK_ALL_TO_ALL_H_

#include <mutex>
#include "arrow_all_to_all.hpp"
namespace cylon {

class LogicalTaskPlan {

 private:
  const std::vector<int> &task_source;
  const std::vector<int> &task_targets;
  const std::vector<int> &worker_sources;
  const std::vector<int> &worker_targets;
  const std::unordered_map<int, int> &task_to_worker;

 public:
  LogicalTaskPlan(const vector<int> &task_source,
                  const vector<int> &task_targets,
                  const vector<int> &worker_sources,
                  const vector<int> &worker_targets,
                  const unordered_map<int,
                                      int> &task_to_worker,
                  const vector<int> &task_source_1);

  const vector<int> &GetTaskSource() const;
  const vector<int> &GetTaskTargets() const;
  const vector<int> &GetWorkerSources() const;
  const vector<int> &GetWorkerTargets() const;
  const unordered_map<int, int> &GetTaskToWorker() const;
};

class ArrowTaskCallBack : public ArrowCallback {
  bool onReceive(int worker_source, std::shared_ptr<arrow::Table> table, int target_task) override;

  virtual bool onReceive(std::shared_ptr<arrow::Table> table, int target) = 0;
};

class ArrowTaskAllToAll : public ArrowAllToAll {

 protected:
  std::mutex mutex;
  const LogicalTaskPlan &plan;

 public:
  ArrowTaskAllToAll(cylon::CylonContext *ctx,
                    const LogicalTaskPlan &plan,
                    int edgeId,
                    const std::shared_ptr<ArrowTaskCallBack> &callback,
                    const std::shared_ptr<arrow::Schema> &schema);

  int InsertTable(const std::shared_ptr<arrow::Table> &arrow, int32_t task_target);
};
}

#endif //CYLON_SRC_CYLON_ARROW_ARROW_TASK_ALL_TO_ALL_H_

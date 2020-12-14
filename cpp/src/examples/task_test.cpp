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

#include <glog/logging.h>
#include <arrow/arrow_task_all_to_all.h>
#include <net/mpi/mpi_communicator.hpp>
#include <thread>
#include <table.hpp>

int main(int argc, char *argv[]) {

  auto config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(config);

  auto tasks_sources = std::make_shared<std::vector<int>>();
  auto tasks_targets = std::make_shared<std::vector<int>>();

  auto worker_sources = std::make_shared<std::vector<int>>();
  auto worker_targets = std::make_shared<std::vector<int>>();

  auto task_to_worker = std::make_shared<std::unordered_map<int, int>>();

  for (int i = 0; i < ctx->GetWorldSize() * 2; i++) {
    tasks_sources->push_back(i);
    tasks_targets->push_back(i);

    LOG(INFO) << "task " << i << "will be on worker " << (i / 2);
    task_to_worker->insert(std::pair<int32_t, int32_t>(i, i / 2));
  }

  for (int i = 0; i < ctx->GetWorldSize(); i++) {
    worker_targets->push_back(i);
    worker_sources->push_back(i);
  }

  auto plan = cylon::LogicalTaskPlan(tasks_sources,
                                     tasks_targets, worker_sources, worker_targets, task_to_worker);

  cylon::ArrowTaskCallBack cb = [&ctx](const std::shared_ptr<arrow::Table> &table, int target) {
    LOG(INFO) << ctx->GetRank() << " received a table to target " << target;
    return true;
  };

  arrow::SchemaBuilder builder;
  auto status = builder.AddField(std::make_shared<arrow::Field>("col1", arrow::int64()));
  status = builder.AddField(std::make_shared<arrow::Field>("col2", arrow::int64()));

  auto schema = builder.Finish();

  cylon::ArrowTaskAllToAll all(ctx, plan, 0, cb, schema.ValueOrDie());

  std::vector<std::shared_ptr<std::thread>> tasks;
  for (int t = ctx->GetRank() * 2; t < ctx->GetRank() * 2 + 2; t++) {
    LOG(INFO) << "Starting  task " << t << " on worker " << ctx->GetRank();
    auto t1 = std::make_shared<std::thread>([ctx, argv, &all, task_to_worker]() {
      std::shared_ptr<cylon::Table> table;

      // todo this can be a mistake
      auto ctx_non_shared = std::make_shared<cylon::CylonContext>(*ctx);
      cylon::FromCSV(ctx_non_shared, argv[1], table);
      LOG(INFO) << "Read table with rows " << table->Rows();

      std::shared_ptr<arrow::Table> arTable;
      table->ToArrowTable(arTable);
      for (auto p:*task_to_worker) {
        LOG(INFO) << ctx->GetRank() << " Sending a table to target " << p.first;
        all.InsertTable(arTable, p.first);
      }
    });
    tasks.push_back(t1);
  }

  auto progress_thread = std::thread([&all]() {
    all.WaitForCompletion();
  });

  for (auto &task:tasks) {
    task->join();
  }

  progress_thread.join();

  ctx->Finalize();
}
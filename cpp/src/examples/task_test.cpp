#include <glog/logging.h>
#include <arrow/arrow_task_all_to_all.h>
#include <net/mpi/mpi_communicator.hpp>
#include <thread>
#include <table.hpp>

int main(int argc, char *argv[]) {

  auto config = cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(&config);

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

  class CallBack : public cylon::ArrowTaskCallBack {

    cylon::CylonContext *ctx;

   public:
    explicit CallBack(cylon::CylonContext *ctx) : ctx(ctx) {
    }

    bool onReceive(const std::shared_ptr<arrow::Table> &table, int target) override {
      LOG(INFO) << this->ctx->GetRank() << " received a table to target " << target;
      return true;
    }
  };

  auto cb = std::make_shared<CallBack>(ctx);

  arrow::SchemaBuilder builder;
  auto status = builder.AddField(std::make_shared<arrow::Field>("col1", arrow::int64()));
  status = builder.AddField(std::make_shared<arrow::Field>("col2", arrow::int64()));

  auto schema = builder.Finish();

  cylon::ArrowTaskAllToAll all(ctx, plan, 0, cb, schema.ValueOrDie());

  std::vector<std::shared_ptr<std::thread>> tasks;
  for (int t = ctx->GetRank() * 2; t < ctx->GetRank() * 2 + 2; t++) {
    LOG(INFO) << "Starting  task " << t << " on worker " << ctx->GetRank();
    auto t1 = std::make_shared<std::thread>([ctx, &all, task_to_worker]() {
      std::shared_ptr<cylon::Table> table;
      cylon::Table::FromCSV(ctx, "/home/chathura/Code/twisterx/cpp/data/csv1.csv", table);
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
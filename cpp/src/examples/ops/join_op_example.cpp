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
#include <chrono>

#include <net/mpi/mpi_communicator.hpp>
#include <ctx/cylon_context.hpp>
#include <table.hpp>
#include <ops/dis_join_op.hpp>
#include <string>

void create_int64_table(char *const *argv,
                        std::shared_ptr<cylon::CylonContext> &ctx,
                        arrow::MemoryPool *pool,
                        std::shared_ptr<arrow::Table> &left_table,
                        std::shared_ptr<arrow::Table> &right_table);

int main(int argc, char *argv[]) {
  if (argc < 3) {
    LOG(ERROR) << "./join_op_example num_tuples_per_worker 0.0-1.0" << std::endl
               << "./join_op_example num_tuples_per_worker 0.0-1.0  [hash | sort]" << std::endl
               << "./join_op_example num_tuples_per_worker 0.0-1.0  csv_file1 csv_file2" << std::endl
               << "./join_op_example num_tuples_per_worker 0.0-1.0  [hash | sort] csv_file1 csv_file2" << std::endl;
    return 1;
  }

  auto start_start = std::chrono::steady_clock::now();
  auto mpi_config = cylon::net::MPIConfig::Make();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  arrow::MemoryPool *pool = arrow::default_memory_pool();
  std::shared_ptr<cylon::Table> first_table, second_table, joined;
  cylon::join::config::JoinAlgorithm algorithm = cylon::join::config::JoinAlgorithm::SORT;
  if (argc >= 3 && argc <= 4) {
    if (argc == 4) {
      if (!strcmp(argv[3], "hash")) {
        algorithm = cylon::join::config::JoinAlgorithm::HASH;
      }
    }

    std::shared_ptr<arrow::Table> left_table, right_table;
    create_int64_table(argv, ctx, pool, left_table, right_table);
    ctx->Barrier();

    auto status = cylon::Table::FromArrowTable(ctx, left_table, first_table);
    if (!status.is_ok()) {
      LOG(INFO) << "Table reading failed " << argv[1];
      ctx->Finalize();
      return 1;
    }

    status = cylon::Table::FromArrowTable(ctx, right_table, second_table);
    if (!status.is_ok()) {
      LOG(INFO) << "Table reading failed " << argv[2];
      ctx->Finalize();
      return 1;
    }
  } else if (argc >= 5 && argc <= 6) {
    if (argc == 6) {
      if (!strcmp(argv[3], "hash")) {
        algorithm = cylon::join::config::JoinAlgorithm::HASH;
      }
      cylon::FromCSV(ctx, std::string(argv[4]) + std::to_string(ctx->GetRank()) + ".csv", first_table);
      cylon::FromCSV(ctx, std::string(argv[5]) + std::to_string(ctx->GetRank()) + ".csv", second_table);
    } else if (argc == 5) {
      cylon::FromCSV(ctx, std::string(argv[4]) + std::to_string(ctx->GetRank()) + ".csv", first_table);
      cylon::FromCSV(ctx, std::string(argv[5]) + std::to_string(ctx->GetRank()) + ".csv", second_table);
    }
  }

  auto read_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Read tables in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                read_end_time - start_start).count() << "[ms]";

  first_table->retainMemory(false);
  second_table->retainMemory(false);

  const cylon::ResultsCallback &callback = [&](int tag, const std::shared_ptr<cylon::Table> &table) {
    LOG(INFO) << tag << " Result received " << table->Rows();
  };


  const auto &join_config = cylon::join::config::JoinConfig::InnerJoin(0, 0, algorithm);
  const auto &part_config = cylon::PartitionOpConfig(ctx->GetWorldSize(), {0});
  const auto &dist_join_config = cylon::DisJoinOpConfig(part_config, join_config);

  auto op = cylon::DisJoinOP(ctx, first_table->get_table()->schema(), 0, callback, dist_join_config);

  op.InsertTable(100, first_table);
  op.InsertTable(200, second_table);
  first_table.reset();
  second_table.reset();
  auto execution = op.GetExecution();
  execution->WaitForCompletion();
  auto join_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "Join done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                join_end_time - read_end_time).count() << "[ms]";
  ctx->Finalize();
  return 0;
}

void create_int64_table(char *const *argv,
                        std::shared_ptr<cylon::CylonContext> &ctx,
                        arrow::MemoryPool *pool,
                        std::shared_ptr<arrow::Table> &left_table,
                        std::shared_ptr<arrow::Table> &right_table) {
  arrow::Int64Builder left_id_builder(pool), right_id_builder(pool);
  arrow::DoubleBuilder cost_builder(pool);
  uint64_t count = std::stoull(argv[1]);
  double dup = std::stod(argv[2]);
  std::random_device rd;
  std::mt19937_64 gen(0);
  std::uniform_int_distribution<int64_t> distrib(0, (int64_t) (count * dup * ctx->GetWorldSize()));

  std::mt19937_64 gen1(rd());
  std::uniform_real_distribution<double> val_distrib;

  arrow::Status st = left_id_builder.Reserve(count);
  st = right_id_builder.Reserve(count);
  st = cost_builder.Reserve(count);
  for (uint64_t i = 0; i < count; i++) {
    int64_t l = distrib(gen);
    int64_t r = distrib(gen);
    double v = val_distrib(gen1);
    left_id_builder.UnsafeAppend(l);
    right_id_builder.UnsafeAppend(r);
    cost_builder.UnsafeAppend(v);
  }

  std::shared_ptr<arrow::Array> left, right, vals;
  st = left_id_builder.Finish(&left);
  st = right_id_builder.Finish(&right);
  st = cost_builder.Finish(&vals);
  std::vector<std::shared_ptr<arrow::Field>> left_schema_vector = {arrow::field("first", left->type()),
                                                              arrow::field("second", vals->type())};
  auto left_schema = std::make_shared<arrow::Schema>(left_schema_vector);
  left_table = arrow::Table::Make(left_schema, {std::move(left), vals});
  std::vector<std::shared_ptr<arrow::Field>> right_schema_vector = {arrow::field("first", right->type()),
                                                               arrow::field("second", vals->type())};
  auto right_schema = std::make_shared<arrow::Schema>(right_schema_vector);
  right_table = arrow::Table::Make(right_schema, {std::move(right), std::move(vals)});
}


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
#include "../example_utils.hpp"

int main(int argc, char *argv[]) {
  if (argc < 4) {
    LOG(ERROR) << "./join_op_example m num_tuples_per_worker 0.0-1.0" << std::endl
               << "./join_op_example m num_tuples_per_worker 0.0-1.0  [hash | sort]" << std::endl
               << "./join_op_example f csv_file1 csv_file2" << std::endl
               << "./join_op_example f csv_file1 csv_file2 [hash | sort] " << std::endl;
    return 1;
  }

  auto start_start = std::chrono::steady_clock::now();
  auto mpi_config = cylon::net::MPIConfig::Make();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  std::shared_ptr<cylon::Table> first_table, second_table, joined;
  cylon::join::config::JoinAlgorithm algorithm = cylon::join::config::JoinAlgorithm::SORT;
  std::string mem = std::string(argv[1]);
  if (mem == "m") {
    if (argc == 5) {
      if (!strcmp(argv[4], "hash")) {
        LOG(INFO) << "Hash join algorithm";
        algorithm = cylon::join::config::JoinAlgorithm::HASH;
      } else {
        LOG(INFO) << "Sort join algorithm";
      }
    } else {
      LOG(INFO) << "Sort join algorithm";
    }
    uint64_t count = std::stoull(argv[2]);
    double dup = std::stod(argv[3]);
    cylon::examples::create_two_in_memory_tables(count, dup,ctx,first_table,second_table);
  } else if (mem == "f") {
    cylon::FromCSV(ctx, std::string(argv[2]) + std::to_string(ctx->GetRank()) + ".csv", first_table);
    cylon::FromCSV(ctx, std::string(argv[3]) + std::to_string(ctx->GetRank()) + ".csv", second_table);
    if (argc == 5) {
      if (!strcmp(argv[4], "hash")) {
        LOG(INFO) << "Hash join algorithm";
        algorithm = cylon::join::config::JoinAlgorithm::HASH;
      } else {
        LOG(INFO) << "Sort join algorithm";
      }
    }
  }

  ctx->Barrier();
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




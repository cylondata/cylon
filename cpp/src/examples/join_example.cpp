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

#include <cylon/ops.hpp>
#include <cylon/net/mpi/mpi_communicator.hpp>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/table.hpp>

#include "example_utils.hpp"


int main(int argc, char *argv[]) {
  if (argc < 5) {
    LOG(ERROR) << "./join_example m [n | o] num_tuples_per_worker 0.0-1.0" << std::endl
               << "./join_example m [n | o] num_tuples_per_worker 0.0-1.0" << std::endl
               << "./join_example f [n | o] csv_file1 csv_file2" << std::endl
               << "./join_example f [n | o] csv_file1 csv_file2" << std::endl;
    return 1;
  }

  auto start_start = std::chrono::steady_clock::now();
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  std::shared_ptr<cylon::CylonContext> ctx;
  if (!cylon::CylonContext::InitDistributed(mpi_config, &ctx).is_ok()) {
    std::cerr << "ctx init failed! " << std::endl;
    return 1;
  }

  std::shared_ptr<cylon::Table> first_table, second_table, joined;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);
  cylon::join::config::JoinAlgorithm algorithm = cylon::join::config::JoinAlgorithm::SORT;

  std::string mem = std::string(argv[1]);
  std::string ops_param = std::string(argv[2]);

  bool ops = true;
  if (ops_param == "o") {
    ops = true;
  } else if (ops_param == "n") {
    ops = false;
  }

  if (mem == "m") {
    if (argc == 6) {
      if (!strcmp(argv[5], "hash")) {
        LOG(INFO) << "Hash join algorithm";
        algorithm = cylon::join::config::JoinAlgorithm::HASH;
      } else {
        LOG(INFO) << "Sort join algorithm";
      }
    } else {
      LOG(INFO) << "Sort join algorithm";
    }
    uint64_t count = std::stoull(argv[3]);
    double dup = std::stod(argv[4]);
    cylon::examples::create_two_in_memory_tables(count, dup,ctx,first_table,second_table);
  } else if (mem == "f") {
    cylon::FromCSV(ctx, std::string(argv[3]) + std::to_string(ctx->GetRank()) + ".csv", first_table);
    cylon::FromCSV(ctx, std::string(argv[4]) + std::to_string(ctx->GetRank()) + ".csv", second_table);

    if (argc == 6) {
      if (!strcmp(argv[5], "hash")) {
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

  auto join_config = cylon::join::config::JoinConfig(cylon::join::config::JoinType::INNER,
                                                     0,
                                                     0,
                                                     algorithm,
                                                     "l_",
                                                     "r_");
  cylon::Status status;
  if (ops) {
    status = cylon::JoinOperation(ctx, first_table, second_table, join_config, joined);
  } else {
    status = cylon::DistributedJoin(first_table, second_table, join_config, joined);
  }
  if (!status.is_ok()) {
    LOG(INFO) << "Table join failed ";
    ctx->Finalize();
    return 1;
  }
  auto join_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "First table had : " << first_table->Rows() << " and Second table had : "
            << second_table->Rows() << ", Joined has : " << joined->Rows();
  LOG(INFO) << "Join done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                join_end_time - read_end_time).count() << "[ms]";
  std::vector<std::string> column_names = joined->ColumnNames();

  ctx->Finalize();
  return 0;
}

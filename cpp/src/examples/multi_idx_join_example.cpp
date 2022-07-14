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

#include <cylon/net/mpi/mpi_communicator.hpp>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/table.hpp>

#include "example_utils.hpp"

#define EXIT_IF_FAILED(expr)    \
  do{                           \
    const auto& _st = (expr);   \
    if (!_st.is_ok()) {         \
      LOG(INFO) << "Fail: " << _st.get_msg(); \
      ctx->Finalize();          \
      return 1;                 \
    };                          \
  } while (0)

int main(int argc, char *argv[]) {
  if (argc < 6) {
    LOG(ERROR) << "./multi_idx_join_example m [n | o] num_tuples_per_worker 0.0-1.0" << std::endl
               << "./multi_idx_join_example m [n | o] num_tuples_per_worker 0.0-1.0" << std::endl
               << "./multi_idx_join_example f [n | o] csv_file1 csv_file2" << std::endl
               << "./multi_idx_join_example f [n | o] csv_file1 csv_file2" << std::endl;
    return 1;
  }

  auto start_start = std::chrono::steady_clock::now();
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

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

  CYLON_UNUSED(ops);

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
            << std::chrono::duration_cast<std::chrono::milliseconds>(read_end_time - start_start).count() << "[ms]";

  auto join_config = cylon::join::config::JoinConfig(cylon::join::config::JoinType::INNER,
                                                     {0, 1},
                                                     {0, 1},
                                                     algorithm,
                                                     "l_",
                                                     "r_");

  EXIT_IF_FAILED(cylon::DistributedJoin(first_table, second_table, join_config, joined));

  auto join_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "First table had : " << first_table->Rows() << " and Second table had : "
            << second_table->Rows() << ", Joined has : " << joined->Rows();
  LOG(INFO) << "Join done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(join_end_time - read_end_time).count() << "[ms]";

  auto column_names = joined->ColumnNames();

  for (const auto &col_name : column_names) {
    std::cout << col_name << ", ";
  }
  std::cout << std::endl;
  joined->Print();

  ctx->Finalize();
  return 0;
}

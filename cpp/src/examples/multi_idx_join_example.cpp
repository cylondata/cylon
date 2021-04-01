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
  if (argc < 3) {
    LOG(ERROR) << "There should be two arguments with paths to csv files";
    return 1;
  }

  auto start_start = std::chrono::steady_clock::now();
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  std::shared_ptr<cylon::Table> first_table, second_table, joined;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);
  EXIT_IF_FAILED(cylon::FromCSV(ctx, argv[1], first_table, read_options));
  EXIT_IF_FAILED(cylon::FromCSV(ctx, argv[2], second_table, read_options));

  auto read_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "Read tables in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(read_end_time - start_start).count() << "[ms]";

  auto join_config = cylon::join::config::JoinConfig(cylon::join::config::JoinType::INNER,
                                                     {0, 1},
                                                     {0, 1},
                                                     cylon::join::config::JoinAlgorithm::HASH,
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

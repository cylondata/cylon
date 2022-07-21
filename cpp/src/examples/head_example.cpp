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
  if ((argc < 5 && std::string(argv[1])  == "f")) {
    LOG(ERROR) << "./head_example f [n | o] csv_file num_rows" << std::endl
               << "./head_example f [n | o] csv_file num_rows" << std::endl;
    return 1;
  }

  if ((argc < 6 && std::string(argv[1]) == "m")) {
    LOG(ERROR) << "./head_example m [n | o] num_tuples_per_worker 0.0-1.0 num_rows" << std::endl
               << "./head_example m [n | o] num_tuples_per_worker 0.0-1.0 num_rows" << std::endl;
    return 1;
  }

  auto start_start = std::chrono::steady_clock::now();
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  std::shared_ptr<cylon::CylonContext> ctx;
  if (!cylon::CylonContext::InitDistributed(mpi_config, &ctx).is_ok()) {
    std::cerr << "ctx init failed! " << std::endl;
    return 1;
  }

  std::shared_ptr<cylon::Table> in_table, head_table;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);
  cylon::join::config::JoinAlgorithm algorithm = cylon::join::config::JoinAlgorithm::SORT;

  std::string mem = std::string(argv[1]);
  std::string ops_param = std::string(argv[2]);
  int64_t num_rows = 0;

  bool ops = true;
  if (ops_param == "o") {
    ops = true;
  } else if (ops_param == "n") {
    ops = false;
  }

  if (mem == "m") {
    uint64_t count = std::stoull(argv[3]);
    double dup = std::stod(argv[4]);
    cylon::examples::create_in_memory_tables(count, dup,ctx,in_table);
    num_rows = std::stoull(argv[5]);
  } else if (mem == "f") {
    LOG(INFO) << "Load From the CSV file" << std::string(argv[3]);
    cylon::FromCSV(ctx, std::string(argv[3]), in_table);

    num_rows = std::stoull(argv[4]);
  }
  ctx->Barrier();
  auto read_end_time = std::chrono::steady_clock::now();
  //in_table->Print();
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

  //Code block for slice operation

  if (ops) {
    status = cylon::Head(in_table, num_rows, head_table);
  } else {
    status = cylon::DistributedHead(in_table, num_rows, head_table);
  }
  if (!status.is_ok()) {
    LOG(INFO) << "Table Head is failed ";
    ctx->Finalize();
    return 1;
  }
  auto slice_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Head table has : " << head_table->Rows();
  LOG(INFO) << "Head is done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                slice_end_time - read_end_time).count() << "[ms]";
  std::vector<std::string> sliced_column_names = head_table->ColumnNames();

  head_table->Print();
  ctx->Finalize();
  return 0;
}

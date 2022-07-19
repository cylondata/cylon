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
  if ((argc < 6 && std::string(argv[1])  == "f")) {
    LOG(ERROR) << "./slice_example f [n | o] csv_file offset length" << std::endl
               << "./slice_example f [n | o] csv_file  offset length" << std::endl;
    return 1;
  }

  if ((argc < 7 && std::string(argv[1]) == "m")) {
    LOG(ERROR) << "./slice_example m [n | o] num_tuples_per_worker 0.0-1.0 offset length" << std::endl
               << "./slice_example m [n | o] num_tuples_per_worker 0.0-1.0 offset length" << std::endl;
    return 1;
  }
  LOG(INFO) << "Starting main() function";
  auto start_start = std::chrono::steady_clock::now();
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  
  std::shared_ptr<cylon::CylonContext> ctx;
  if (!cylon::CylonContext::InitDistributed(mpi_config, &ctx).is_ok()) {
    std::cerr << "ctx init failed! " << std::endl;
    return 1;
  }


  std::shared_ptr<cylon::Table> in_table, joined, sliced;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);
  cylon::join::config::JoinAlgorithm algorithm = cylon::join::config::JoinAlgorithm::SORT;

  std::string mem = std::string(argv[1]);
  std::string ops_param = std::string(argv[2]);
  int64_t offset = 0, length = 0;

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
    offset = std::stoull(argv[5]);
    length = std::stoull(argv[6]);
    LOG(INFO) << "Load From in-memory size: " << std::string(argv[3]);
  } else if (mem == "f") {
    LOG(INFO) << "Load From the CSV file: " << std::string(argv[3]);
    cylon::FromCSV(ctx, std::string(argv[3]) , in_table);

    //cylon::FromCSV(ctx, std::string(argv[3]) + std::to_string(ctx->GetRank()) + ".csv", first_table);
    //cylon::FromCSV(ctx, std::string(argv[4]) + std::to_string(ctx->GetRank()) + ".csv", second_table);

    offset = std::stoull(argv[4]);
    length = std::stoull(argv[5]);
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

  // Arup: Code block for slice operation
  int order = 0;
  if (ops) {
    status = cylon::Local_Slice(in_table, offset, length, sliced);
  } else {
    status = cylon::Distributed_Slice(in_table, offset, length, sliced, order);
  }
  if (!status.is_ok()) {
    LOG(INFO) << "Table Slice is failed ";
    ctx->Finalize();
    return 1;
  }
  auto slice_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Sliced table has : " << sliced->Rows();
  LOG(INFO) << "Sliced is done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                slice_end_time - read_end_time).count() << "[ms]";
  std::vector<std::string> sliced_column_names = sliced->ColumnNames();

  sliced->Print();
  sleep(3);
  ctx->Finalize();
  return 0;
}

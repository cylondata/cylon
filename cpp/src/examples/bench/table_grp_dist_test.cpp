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

#include <net/mpi/mpi_communicator.hpp>
#include <ctx/cylon_context.hpp>
#include <table.hpp>
#include <status.hpp>
#include <iostream>
#include <io/csv_read_config.hpp>
#include <chrono>
#include <groupby/groupby.hpp>

using namespace cylon;
using namespace cylon::join::config;

bool Run(int iter, int rank,
         std::shared_ptr<Table> &table1,
         std::shared_ptr<Table> &output) {
  Status status;

  auto t1 = std::chrono::high_resolution_clock::now();
  status = cylon::GroupBy(table1, 0, {1},{GroupByAggregationOp::SUM}, output);
  auto t2 = std::chrono::high_resolution_clock::now();
  table1->GetContext()->Barrier();
  auto t3 = std::chrono::high_resolution_clock::now();

  if (status.is_ok()) {
    LOG(INFO) << iter << " " << table1->GetContext()->GetWorldSize() << " " << rank << " groupby "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " w_t " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
              << " tot " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t1).count()
              << " res " << output->Rows();
    output->Clear();
    return true;
  } else {
    LOG(ERROR) << "Join write failed!";
    output->Clear();
    return false;
  }
}

int main(int argc, char *argv[]) {

  std::shared_ptr<Table> table1, out;
  Status status;

  auto mpi_config = cylon::net::MPIConfig::Make();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  int rank = ctx->GetRank();
  std::string srank = std::to_string(rank);

  if (argc != 4) {
    LOG(ERROR) << "src_dir and base_dir iter not provided! ";
    return 1;
  }

  std::string src_dir = argv[1];
  std::string base_dir = argv[2];
  int iter = std::stoi(argv[3]);

  system(("mkdir -p " + base_dir).c_str());

  std::string csv1 = base_dir + "/csv1_" + srank + ".csv";

  system(("cp " + src_dir + "/csv1_" + srank + ".csv " + csv1).c_str());

  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);
  if (!(status = cylon::FromCSV(ctx, csv1, table1, read_options)).is_ok()) {
    LOG(ERROR) << "File read failed! " << csv1;
    return 1;
  }
  ctx->Barrier();

  for (int i = 0; i < iter; i++) {
    Run(i, rank, table1, out);
  }

  ctx->Finalize();

  system(("rm " + csv1).c_str());

  return 0;
}



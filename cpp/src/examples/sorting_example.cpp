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

int main(int argc, char *argv[]) {
  if (argc < 3) {
    LOG(ERROR) << "./sorting_example m num_tuples_per_worker 0.0-1.0" << std::endl
               << "./sorting_example f csv_file1" << std::endl;
    return 1;
  }

  auto start_start = std::chrono::steady_clock::now();
  CYLON_UNUSED(start_start);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  std::shared_ptr<cylon::Table> first_table, output;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  std::string mem = std::string(argv[1]);
  if (mem == "m") {
    uint64_t count = std::stoull(argv[2]);
    double dup = std::stod(argv[3]);
    cylon::examples::create_in_memory_tables(count, dup, ctx, first_table);
  } else if (mem == "f") {
    cylon::FromCSV(ctx, std::string(argv[2]) + std::to_string(ctx->GetRank()) + ".csv", first_table);
  }
  ctx->Barrier();
  auto status = cylon::DistributedSort(first_table, 0, output);
  if (status.is_ok() && output->Rows() <= 1000) {
    output->Print();
  }
  ctx->Finalize();
  return 0;
}

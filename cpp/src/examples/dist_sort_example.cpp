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



/**
 * To run this example: ./dist_sort_example <num_rows_per_worker> <num_iterations>
 */

#include <glog/logging.h>
#include <chrono>

#include <cylon/net/mpi/mpi_communicator.hpp>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/table.hpp>

#include "example_utils.hpp"

int64_t run_example(std::shared_ptr<cylon::Table>& table) {
  std::shared_ptr<cylon::Table> output;
  auto start_1 = std::chrono::steady_clock::now();
  auto status = DistributedSort(table, 0, output, true);
  auto end_1 = std::chrono::steady_clock::now();

  int64_t time = std::chrono::duration_cast<std::chrono::milliseconds>(end_1 - start_1).count();

  return time;
}

int main(int argc, char *argv[]) {
  if (argc < 1) {
    LOG(ERROR) << "./dist_sort_example <num_per_worker> <num_iterations>" << std::endl;
    return 1;
  }

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  std::shared_ptr<cylon::CylonContext> ctx;
  auto status = cylon::CylonContext::InitDistributed(mpi_config, &ctx);

  std::shared_ptr<cylon::Table> table, output_merge, output_sort;

  uint64_t count = std::stoull(argv[1]);
  cylon::examples::create_in_memory_tables(count, 0.2, ctx, table);

  int iters = std::stoi(argv[2]);

  ctx->Barrier();

  int64_t time = 0;

  for(int i = 0; i < iters; i++) {
    time += run_example(table);
  }

  std::cout<< time << " ms in total." << std::endl;

  return 0;
}

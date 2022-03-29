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
  if (argc < 1) {
    LOG(ERROR) << "./dist_sort_example <num_per_worker>" << std::endl;
    return 1;
  }

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  std::shared_ptr<cylon::Table> table, output_merge, output_sort;

  uint64_t count = std::stoull(argv[1]);
  cylon::examples::create_in_memory_tables(count, 0.2, ctx, table);

  auto start_2 = std::chrono::steady_clock::now();
  auto status = DistributedSort(table, 0, output_sort, true,
                    {0, 0, cylon::SortOptions::REGULAR_SAMPLE_SORT});
  auto end_2 = std::chrono::steady_clock::now();

  LOG(INFO)<< "using sort takes " 
  << std::chrono::duration_cast<std::chrono::milliseconds>(end_2 - start_2).count()
  << " ms." << std::endl;
  
  auto start_1 = std::chrono::steady_clock::now();
  status = DistributedSort(table, 0, output_merge, true,
                    {0, 0, cylon::SortOptions::REGULAR_SAMPLE_MERGE});
  auto end_1 = std::chrono::steady_clock::now();

  LOG(INFO)<< "using merge takes " 
    << std::chrono::duration_cast<std::chrono::milliseconds>(end_1 - start_1).count()
    << " ms." << std::endl;
  
  ctx->Finalize();
  return 0;
}

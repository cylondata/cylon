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

#include <iostream>
#include <fstream>
#include <string>

#include <glog/logging.h>
#include <chrono>
#include <thread>

#include <cudf/table/table.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/types.hpp>

#include <gcylon/gtable_api.hpp>
#include <gcylon/utils/construct.hpp>
#include <examples/gcylon/print.hpp>
#include <cylon/net/mpi/mpi_communicator.hpp>

using namespace gcylon;
using namespace std::chrono;

void writeToFile(cudf::table_view tv, std::string file_name, int my_rank) {
  cudf::io::sink_info sinkInfo(file_name);
  cudf::io::csv_writer_options writerOptions = cudf::io::csv_writer_options::builder(sinkInfo, tv);
  cudf::io::write_csv(writerOptions);
  LOG(INFO) << my_rank << ", written the table to the file: " << file_name;
}

int main(int argc, char *argv[]) {

  const int COLS = 4;
  const int ROWS = 20;
  const bool RESULT_TO_FILE = true;

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
  int my_rank = ctx->GetRank();

  int number_of_GPUs;
  cudaGetDeviceCount(&number_of_GPUs);

  // set the gpu
  cudaSetDevice(my_rank % number_of_GPUs);
  int deviceInUse = -1;
  cudaGetDevice(&deviceInUse);
  LOG(INFO) << "my_rank: " << my_rank << ", device in use: " << deviceInUse << ", number of GPUs: " << number_of_GPUs;

  std::shared_ptr<cudf::table> tbl = constructRandomDataTable(COLS, ROWS, ctx->GetRank() * 100);
  auto tv = tbl->view();
  LOG(INFO) << "my_rank: " << my_rank << ", initial dataframe. cols: " << tv.num_columns() << ", rows: "
            << tv.num_rows();
  if (my_rank == 0) {
    LOG(INFO) << "my_rank: " << my_rank << ", initial dataframe................................. ";
    printLongTable(tv);
  }

  // sort the table
  std::vector<cudf::size_type> sort_columns = {0, 1};
  std::vector<cudf::order> column_orders{sort_columns.size(), cudf::order::ASCENDING};
  std::unique_ptr<cudf::table> sorted_table;

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  DistributedSort(tv, sort_columns, column_orders, ctx, sorted_table);
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double, std::milli> diff = t2 - t1;
  long int delay = diff.count();

  LOG(INFO) << "my_rank: " << my_rank << ", duration: " << delay;
  auto sorted_tv = sorted_table->view();
  if (my_rank == 0) {
    LOG(INFO) << "my_rank: " << my_rank << ", sorted dataframe................................. ";
    printLongTable(sorted_tv);
  }
  LOG(INFO) << "my_rank: " << my_rank << ", rows in sorted table: " << sorted_tv.num_rows();

  if (RESULT_TO_FILE) {
    std::string file_name = std::string("sorted_table_") + std::to_string(ctx->GetRank()) + ".csv";
    writeToFile(sorted_tv, file_name, ctx->GetRank());
  }

  ctx->Finalize();
  return 0;
}

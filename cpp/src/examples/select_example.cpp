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


/**
 * This example reads a csv file and selects few records from it based on a function
 */
int main(int argc, char *argv[]) {
  if (argc < 2) {
    LOG(ERROR) << "There should be an argument with path to a csv file";
    return 1;
  }
  auto start_time = std::chrono::steady_clock::now();
  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  std::shared_ptr<cylon::Table> table, select;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(
      false).BlockSize(1 << 30);

  auto status = cylon::Table::FromCSV(ctx, argv[1], table, read_options);
  auto read_end_time = std::chrono::steady_clock::now();

  if (!status.is_ok()) {
    LOG(ERROR) << "Table reading has failed  : " << status.get_msg();
    ctx->Finalize();
    return 1;
  }
  LOG(INFO) << "Read table in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                read_end_time - start_time).count() << "[ms]";
  status = table->Select([](cylon::Row row) {
    return row.GetInt64(0) % 2 == 0;
  }, select);
  auto select_end_time = std::chrono::steady_clock::now();
  if (!status.is_ok()) {
    LOG(ERROR) << "Table select has failed  : " << status.get_msg();
    ctx->Finalize();
    return 1;
  }
  LOG(INFO) << "Table had : " << table->Rows() << ", Select has : " << select->Rows();
  LOG(INFO) << "Select done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                read_end_time - select_end_time).count()
            << "[ms]";
  ctx->Finalize();
  LOG(INFO) << "Operation took : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                select_end_time - start_time).count()
            << "[ms]";
  return 0;
}

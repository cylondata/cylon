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

#include <table.hpp>
#include <net/mpi/mpi_communicator.hpp>

int main(int argc, char *argv[]) {
  auto start_time = std::chrono::steady_clock::now();
  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
  std::shared_ptr<cylon::Table> first_table, second_table, result;

  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);
  auto status = cylon::Table::FromCSV(ctx, argv[1], first_table, read_options);
  if (!status.is_ok()) {
    LOG(INFO) << "Table reading failed " << argv[1];
    ctx->Finalize();
    return 1;
  }

  status = cylon::Table::FromCSV(ctx, argv[2], second_table, read_options);
  if (!status.is_ok()) {
    LOG(INFO) << "Table reading failed " << argv[2];
    ctx->Finalize();
    return 1;
  }
  auto read_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Read tables in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                read_end_time - start_time).count() << "[ms]";
  if (!status.is_ok()) {
    LOG(INFO) << "Table intersection failed";
    ctx->Finalize();
    return 1;
  }

  status = first_table->DistributedSubtract(second_table, result);
  if (!status.is_ok()) {
    LOG(INFO) << "Table intersection failed";
    ctx->Finalize();
    return 1;
  }
  auto end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Table 1 had : " << first_table->Rows()
            << " and Table 2 had : " << second_table->Rows()
            << ", subtract has : " << result->Rows();
  LOG(INFO) << "Subtract done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - read_end_time).count() << "[ms]";
  ctx->Finalize();
  return 0;
}

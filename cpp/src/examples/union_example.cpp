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
 * This example reads two csv files and does a union on them.
 */
int main(int argc, char *argv[]) {
  if (argc < 3) {
    LOG(ERROR) << "There should be two arguments with paths to csv files";
    return 1;
  }

  auto start_time = std::chrono::steady_clock::now();
  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  std::shared_ptr<cylon::Table> first_table, second_table, unioned_table;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  auto status = cylon::Table::FromCSV(ctx, argv[1], first_table, read_options);
  if (!status.is_ok()) {
    LOG(INFO) << "Table reading failed " << argv[1];
    ctx->Finalize();
    return 1;
  }
  // read second table
  status = cylon::Table::FromCSV(ctx, argv[2], second_table, read_options);
  if (!status.is_ok()) {
    LOG(INFO) << "Table reading failed " << argv[2];
    ctx->Finalize();
    return 1;
  }
  auto read_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Read all in " << std::chrono::duration_cast<std::chrono::milliseconds>(
      read_end_time - start_time).count() << "[ms]";

  auto union_start_time = std::chrono::steady_clock::now();
  // apply union operation
  status = first_table->DistributedUnion(second_table, unioned_table);
  if (!status.is_ok()) {
    LOG(INFO) << "Union failed " << status.get_msg();
    ctx->Finalize();
    return 1;
  }
  read_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "First table had : " << first_table->Rows() << " and Second table had : "
            << second_table->Rows() << ", Union has : "
            << unioned_table->Rows() << " rows";
  LOG(INFO) << "Union done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                read_end_time - union_start_time).count()
            << "[ms]";

  ctx->Finalize();
  auto end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Operation took : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count() << "[ms]";
  return 0;
}

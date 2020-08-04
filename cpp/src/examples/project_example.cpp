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
 * This example apply the project operation to a csv file
 */
int main(int argc, char *argv[]) {
  if (argc < 2) {
    LOG(ERROR) << "There should be an argument with path to a csv file";
    return 1;
  }
  auto start_time = std::chrono::steady_clock::now();
  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  std::shared_ptr<cylon::Table> table, project;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  auto status = cylon::Table::FromCSV(ctx, argv[1], table, read_options);
  auto read_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Read table in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                read_end_time - start_time).count()
            << "[ms]";
  if (!status.is_ok()) {
    LOG(INFO) << "Table reading has failed  : " << status.get_msg();
    ctx->Finalize();
    return 1;
  }

  status = table->Project({0}, project);
  if (!status.is_ok()) {
    LOG(INFO) << "Project failed  : " << status.get_msg();
    ctx->Finalize();
    return 1;
  }
  auto project_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Table had : " << table->Columns() << "," << table->Rows() << ", Project has : "
            << project->Columns() << "," << project->Rows();
  LOG(INFO) << "Project done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                project_time - read_end_time).count()
            << "[ms]";
  ctx->Finalize();
  return 0;
}

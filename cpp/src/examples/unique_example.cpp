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

int sequential(std::shared_ptr<cylon::Table> &table, std::shared_ptr<cylon::Table> &out, const std::vector<int> &cols);
int distributed(std::shared_ptr<cylon::Table> &table, std::shared_ptr<cylon::Table> &out, const std::vector<int> &cols);

/**
 * This example reads two csv files and does a union on them.
 */
int main() {

  auto start_time = std::chrono::steady_clock::now();
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  std::shared_ptr<cylon::Table> first_table, unique_table;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string file_path = "/tmp/sub_unique_" + std::to_string(ctx->GetRank()) + ".csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << file_path << std::endl;
  auto status = cylon::FromCSV(ctx, file_path, first_table, read_options);
  if (!status.is_ok()) {
    LOG(INFO) << "Table reading failed " << file_path;
    ctx->Finalize();
    return 1;
  }

  auto read_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Read all in " << std::chrono::duration_cast<std::chrono::milliseconds>(
      read_end_time - start_time).count() << "[ms]";

  auto union_start_time = std::chrono::steady_clock::now();
  std::vector<int> cols = {0, 1};

  if (ctx->GetWorldSize() == 1) {
    sequential(first_table, unique_table, cols);
  } else {
    distributed(first_table, unique_table, cols);
  }

  read_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "First table had : " << first_table->Rows()
            << ", Union has : "
            << unique_table->Rows() << " rows";
  LOG(INFO) << "Unique done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                read_end_time - union_start_time).count()
            << "[ms]";

  ctx->Finalize();
  auto end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Operation took : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count() << "[ms]";

  std::cout << " Original Data" << std::endl;

  first_table->Print();

  std::cout << " Unique Data" << std::endl;

  unique_table->Print();

  return 0;
}

int sequential(std::shared_ptr<cylon::Table> &table, std::shared_ptr<cylon::Table> &out, const std::vector<int> &cols) {
  // apply unique operation
  auto ctx = table->GetContext();
  auto status = cylon::Unique(table, cols, out, true);
  if (!status.is_ok()) {
    LOG(INFO) << "Unique failed " << status.get_msg();
    ctx->Finalize();
    return 1;
  }
  return 0;
}

int distributed(std::shared_ptr<cylon::Table> &table,
                std::shared_ptr<cylon::Table> &out,
                const std::vector<int> &cols) {
  auto ctx = table->GetContext();
  auto status = cylon::DistributedUnique(table, cols, out);
  if (!status.is_ok()) {
    LOG(INFO) << "Distributed Unique failed " << status.get_msg();
    ctx->Finalize();
    return 1;
  }
  auto write_opts = cylon::io::config::CSVWriteOptions().WithDelimiter(',');
  cylon::WriteCSV(out, "/tmp/dist_unique_" + std::to_string(ctx->GetRank()), write_opts);
  return 0;
}
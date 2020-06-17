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
#include <net/mpi/mpi_communicator.h>
#include <ctx/twisterx_context.h>
#include <table.hpp>
#include <chrono>

int main(int argc, char *argv[]) {

  auto tstart = std::chrono::steady_clock::now();

  auto mpi_config = new twisterx::net::MPIConfig();
  auto ctx = twisterx::TwisterXContext::InitDistributed(mpi_config);

  std::shared_ptr<twisterx::Table> table1, table2, unioned;

  LOG(INFO) << "Reading tables";
  auto read_options = twisterx::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  auto t1 = std::chrono::steady_clock::now();

  auto status1 = twisterx::Table::FromCSV("/tmp/csv1.csv", &table1, read_options);
  auto t2 = std::chrono::steady_clock::now();
  LOG(INFO) << "Read table 1 in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "[ms]";

  t1 = std::chrono::steady_clock::now();
  auto status2 = twisterx::Table::FromCSV("/tmp/csv2.csv", &table2, read_options);
  t2 = std::chrono::steady_clock::now();

  LOG(INFO) << "Read table 2 in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "[ms]";
  LOG(INFO) << "Done reading tables";

  if (status1.is_ok() && status2.is_ok()) {
    t1 = std::chrono::steady_clock::now();
    twisterx::Status status = table1->DistributedUnion(ctx, table2, unioned);
    t2 = std::chrono::steady_clock::now();

    LOG(INFO) << "Done union tables " << status.get_msg();
    //unioned->print();
    LOG(INFO) << "Table 1 had : " << table1->rows() << " and Table 2 had : " << table2->rows() << ", Union has : "
              << unioned->rows();
    LOG(INFO) << "Union done in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "[ms]";
  } else {
    LOG(INFO) << "Table reading has failed  : " << status1.get_msg() << ":" << status2.get_msg();
  }
  ctx->Finalize();

  auto tend = std::chrono::steady_clock::now();
  LOG(INFO) << "Operation took : " << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count()
            << "[ms]";
  return 0;
}

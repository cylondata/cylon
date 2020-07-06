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
#include <ctx/cylon_context.h>
#include <table.hpp>
#include <chrono>

int main(int argc, char *argv[]) {

  auto tstart = std::chrono::steady_clock::now();

  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  std::shared_ptr<cylon::Table> table1, table2, unioned;

  LOG(INFO) << "Reading tables";
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  auto t1 = std::chrono::steady_clock::now();

  t1 = std::chrono::steady_clock::now();
  std::vector<std::string> paths{"/home/chathura/Code/twisterx/cpp/data/csv2.csv",
      "/home/chathura/Code/twisterx/cpp/data/csv1.csv"};
//  std::vector<std::shared_ptr<cylon::Table> *> tables{&table1, &table2};
  //auto status3 = cylon::Table::FromCSV(ctx, paths, tables, read_options);

  auto status3 = cylon::Table::FromCSV(ctx, argv[0], table1, read_options);
  status3 = cylon::Table::FromCSV(ctx, argv[1], table2, read_options);

  auto t2 = std::chrono::steady_clock::now();
  LOG(INFO) << "Read all in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "[ms]";

  if (status3.is_ok()) {
    t1 = std::chrono::steady_clock::now();
    cylon::Status status = table1->DistributedUnion(table2, unioned);
    t2 = std::chrono::steady_clock::now();

    LOG(INFO) << "Done union tables " << status.get_msg();
    //unioned->print();
    LOG(INFO) << "Table 1 had : " << table1->Rows() << " and Table 2 had : " << table2->Rows() << ", Union has : "
              << unioned->Rows();
    LOG(INFO) << "Union done in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "[ms]";
  } else {
    LOG(INFO) << "Table reading has failed  : " << status3.get_msg();
  }
  ctx->Finalize();

  auto tend = std::chrono::steady_clock::now();
  LOG(INFO) << "Operation took : " << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count()
            << "[ms]";
  return 0;
}

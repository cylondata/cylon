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

int main(int argc, char *argv[]) {

  auto mpi_config = new twisterx::net::MPIConfig();
  auto ctx = twisterx::TwisterXContext::InitDistributed(mpi_config);

  std::shared_ptr<twisterx::Table> table1, table2, unioned;

  LOG(INFO) << "Reading tables";
  auto status1 = twisterx::Table::FromCSV("/home/chathura/Code/twisterx/cpp/data/csv1.csv", &table1);
  auto status2 = twisterx::Table::FromCSV("/home/chathura/Code/twisterx/cpp/data/csv2.csv", &table2);
  LOG(INFO) << "Done reading tables";

  twisterx::Status status = table1->Union(table2, unioned);

  LOG(INFO) << "Done union tables " << status.get_msg();
  //unioned->print();
  LOG(INFO) << "Table 1 had : " << table1->rows() << " and Table 2 had : " << table2->rows() << ", Union has : "
            << unioned->rows();
  ctx->Finalize();
  return 0;
}

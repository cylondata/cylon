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

  std::shared_ptr<twisterx::Table> table1, table2, joined;
  std::string join_file = "/tmp/csv.csv";
  auto status1 = twisterx::Table::FromCSV(join_file, &table1) ;
  auto status2 = twisterx::Table::FromCSV(join_file, &table2);

  table1->DistributedJoin(ctx, table2,
               twisterx::join::config::JoinConfig::InnerJoin(0, 0),
               &joined);
  ctx->Finalize();
  return 0;
}

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
#include <python/twisterx_context_wrap.h>
#include <python/table_cython.h>
#include "util/uuid.hpp"

void dist_join(bool run);

int main(int argc, char *argv[]) {
  dist_join(true);

  return 0;
}

void dist_join(bool run) {
  std::string mpi_config = "mpi";
  twisterx::python::twisterx_context_wrap *ctx_wrap = new twisterx::python::twisterx_context_wrap(mpi_config);
  auto ctx = ctx_wrap->getInstance();
  std::cout << "Hello World , Rank " << ctx_wrap->GetRank() << ", Size " << ctx_wrap->GetWorldSize() << std::endl;

  std::string uuid_l = twisterx::util::uuid::generate_uuid_v4();
  std::string uuid_r = twisterx::util::uuid::generate_uuid_v4();
  auto status1 = twisterx::python::table::CxTable::from_csv(ctx_wrap, "/tmp/csv.csv", ',', uuid_l);
  auto status2 = twisterx::python::table::CxTable::from_csv(ctx_wrap, "/tmp/csv.csv", ',', uuid_r);

  auto tb_l = new twisterx::python::table::CxTable(uuid_l);
  auto tb_r = new twisterx::python::table::CxTable(uuid_r);
  auto join_config = twisterx::join::config::JoinConfig::RightJoin(0, 1);

  tb_l->distributed_join(ctx_wrap, tb_r->get_id(), join_config);

  ctx_wrap->Finalize();
}

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
twisterx::python::twisterx_context_wrap* get_new_context();

int main(int argc, char *argv[]) {
  dist_join(true);

  return 0;
}

twisterx::python::twisterx_context_wrap* get_new_context() {
  std::cout << "Creating New Contet " << std::endl;
  std::string mpi_config = "mpi";
  twisterx::python::twisterx_context_wrap *ctx_wrap = new twisterx::python::twisterx_context_wrap(mpi_config);
  return ctx_wrap;
}

void dist_join(bool run) {
  auto ctx_wrap = get_new_context();
  auto ctx = ctx_wrap->getInstance();

  auto ctx_wrap1 = get_new_context();
  auto ctx1 = ctx_wrap1->getInstance();

  auto ctx_wrap2 = get_new_context();
  auto ctx2 = ctx_wrap2->getInstance();

  std::cout << "Hello World , Rank [ " << ctx_wrap->GetRank() << "," << ctx_wrap1->GetRank() << "," << ctx_wrap2->GetRank() <<  " ], Size " << ctx_wrap->GetWorldSize() << std::endl;

  std::string uuid_l = twisterx::util::uuid::generate_uuid_v4();
  std::string uuid_r = twisterx::util::uuid::generate_uuid_v4();
  auto status1 = twisterx::python::table::CxTable::from_csv("/tmp/csv.csv", ',', uuid_l);
  auto status2 = twisterx::python::table::CxTable::from_csv( "/tmp/csv.csv", ',', uuid_r);

  auto tb_l = new twisterx::python::table::CxTable(uuid_l);
  auto tb_r = new twisterx::python::table::CxTable(uuid_r);
  auto join_config = twisterx::join::config::JoinConfig::RightJoin(0, 1);

  tb_l->distributed_join(tb_r->get_id(), join_config);   

  //tb_l->distributed_join(ctx_wrap, tb_r->get_id(), join_config);

  

  ctx_wrap->Finalize();
}

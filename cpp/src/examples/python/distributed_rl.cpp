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
#include <net/mpi/mpi_communicator.hpp>
#include <python/cylon_context_wrap.h>
#include <python/table_cython.h>
#include "util/uuid.hpp"

void dist_join(bool run);
cylon::python::cylon_context_wrap* get_new_context();

int main(int argc, char *argv[]) {
  dist_join(true);

  return 0;
}

cylon::python::cylon_context_wrap* get_new_context() {
  std::cout << "Creating New Contet " << std::endl;
  std::string mpi_config = "mpi";
  cylon::python::cylon_context_wrap *ctx_wrap = new cylon::python::cylon_context_wrap(mpi_config);
  return ctx_wrap;
}

void dist_join(bool run) {
  auto ctx_wrap = get_new_context();
  auto ctx = ctx_wrap->getInstance();

  auto ctx_wrap1 = get_new_context();
  auto ctx1 = ctx_wrap1->getInstance();

  auto ctx_wrap2 = get_new_context();
  auto ctx2 = ctx_wrap2->getInstance();

  std::cout << "Hello World , Rank [ " << ctx_wrap->GetRank() << "," << ctx_wrap1->GetRank()
            << "," << ctx_wrap2->GetRank() <<  " ], Size " << ctx_wrap->GetWorldSize() << std::endl;

  std::string uuid_l = cylon::util::generate_uuid_v4();
  std::string uuid_r = cylon::util::generate_uuid_v4();
  auto status1 = cylon::python::table::CxTable::from_csv("/tmp/csv.csv", ',', uuid_l);
  auto status2 = cylon::python::table::CxTable::from_csv( "/tmp/csv.csv", ',', uuid_r);

  auto tb_l = new cylon::python::table::CxTable(uuid_l);
  auto tb_r = new cylon::python::table::CxTable(uuid_r);
  auto join_config = cylon::join::config::JoinConfig::RightJoin(0, 1);

  //tb_l->distributed_join(tb_r->get_id(), join_config);   

  //tb_l->distributed_join(ctx_wrap, tb_r->get_id(), join_config);
  std::cout << "Union Sequential" << std::endl;
  tb_l->Union(tb_r->get_id());
  
  std::cout << "Union Distributed" << std::endl;
  tb_l->DistributedUnion(tb_r->get_id());

  std::cout << "Intersect Sequential" << std::endl;
  tb_l->Intersect(tb_r->get_id());

  std::cout << "Intersect Distributed" << std::endl;
  tb_l->DistributedIntersect(tb_r->get_id());

  std::cout << "Subtract Sequential" << std::endl;
  tb_l->Subtract(tb_r->get_id());

  std::cout << "Subtract Distributed" << std::endl;
  tb_l->DistributedSubtract(tb_r->get_id());

  std::cout << "Project" << std::endl;
  tb_l->Project({0});

  ctx_wrap->Finalize();
}

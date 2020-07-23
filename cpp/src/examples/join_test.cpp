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
#include <ctx/cylon_context.hpp>
#include <table.hpp>

#include "test_utils.hpp"

using namespace cylon;

int main(int argc, char *argv[]) {

  Status status;
  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  int rank = ctx->GetRank(), world_sz = ctx->GetWorldSize(), results = 0;
  
  std::string path1 = "/tmp/cylon/input/csv1_" + std::to_string(rank) + ".csv";
  std::string path2 = "/tmp/cylon/input/csv2_" + std::to_string(rank) + ".csv";
  
  std::string out_path = "/tmp/cylon/output/join_inner_" + std::to_string(world_sz) + "_" + std::to_string(rank) + ".csv";
  const join::config::JoinConfig &join_config = join::config::JoinConfig::InnerJoin(0, 0);
  if (test::TestJoinOperation(join_config, ctx, path1, path2, out_path)) {
    LOG(ERROR) << "union failed!";
    results++;
  }

  ctx->Finalize();
  return results;
}

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

#define CATCH_CONFIG_MAIN
#include "common/catch.hpp"

#include <mpi.h>
#include <iostream>

#include "cylon/net/mpi/mpi_communicator.hpp"
#include "cylon/ctx/cylon_context.hpp"
#include "test_utils.hpp"

namespace cylon {
namespace test {

TEST_CASE("custom mpi communicator") {
  MPI_Init(nullptr, nullptr);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  int rank, world_sz;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_sz);
  INFO("global rank:" + std::to_string(rank) + " sz:" + std::to_string(world_sz));

  // test RETURN_CYLON_STATUS_IF_MPI_FAILED macro
  auto failing_dummy = []() {
    RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Comm_split(MPI_COMM_WORLD, 0, 0, nullptr));
    return Status::OK();
  };
  REQUIRE(!failing_dummy().is_ok());

  if (world_sz == 4) {
    // world 3-1 split
    int l_rank = rank < 3 ? rank : rank - 3; // [0, 1, 2, 1]
    int color = rank < 3 ? 0 : 1; // [0, 0, 0, 1]
    MPI_Comm new_comm;
    REQUIRE(MPI_Comm_split(MPI_COMM_WORLD, color, l_rank, &new_comm) == MPI_SUCCESS);

    auto mpi_config = cylon::net::MPIConfig::Make(new_comm);
    auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

    REQUIRE(l_rank == ctx->GetRank());
    if (color == 0) {
      REQUIRE(ctx->GetWorldSize() == 3);
    } else { // color == 1
      REQUIRE(ctx->GetWorldSize() == 1);
    }

    REQUIRE(MPI_Comm_free(&new_comm) == MPI_SUCCESS);
    ctx->Finalize(); // should not finalize MPI here

    // world 2-2 split - join
    l_rank = rank % 2; // [0, 1, 0, 1]
    color = rank / 2; // [0, 0, 1, 1]
    REQUIRE(MPI_Comm_split(MPI_COMM_WORLD, color, l_rank, &new_comm) == MPI_SUCCESS);

    mpi_config = cylon::net::MPIConfig::Make(new_comm);
    ctx = cylon::CylonContext::InitDistributed(mpi_config);

    REQUIRE(l_rank == ctx->GetRank());
    REQUIRE(ctx->GetWorldSize() == 2);

    std::string path1 = "../data/input/csv1_" + std::to_string(l_rank) + ".csv";
    std::string path2 = "../data/input/csv2_" + std::to_string(l_rank) + ".csv";
    std::string out_path =
        "../data/output/join_inner_" + std::to_string(2) + "_" + std::to_string(l_rank) + ".csv";

    SECTION("testing inner joins - sort") {
      const auto &join_config =
          join::config::JoinConfig::InnerJoin(0, 0, cylon::join::config::JoinAlgorithm::SORT);
      test::TestJoinOperation(join_config, ctx, path1, path2, out_path);
    }

    ctx->Finalize(); // should not finalize MPI here
  }

  MPI_Finalize();
}

}
}
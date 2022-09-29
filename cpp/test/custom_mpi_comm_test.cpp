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

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include <mpi.h>
#include <iostream>

#include "cylon/net/mpi/mpi_communicator.hpp"
#include "cylon/ctx/cylon_context.hpp"
#include "test_utils.hpp"

#ifdef BUILD_CYLON_GLOO
#include "cylon/net/gloo/gloo_communicator.hpp"
#endif

#ifdef BUILD_CYLON_UCX
#include "cylon/net/ucx/ucx_communicator.hpp"
#endif

std::string COMM_ARG;

namespace cylon {
namespace test {

std::shared_ptr<net::CommConfig> MakeConfig(MPI_Comm comm){
  if (COMM_ARG == "mpi") {
    LOG(INFO) << "Using MPI";
    return net::MPIConfig::Make(comm);
  }

#ifdef BUILD_CYLON_GLOO
  if (COMM_ARG == "gloo-mpi") {
    LOG(INFO) << "Using Gloo with MPI";
    return net::GlooConfig::MakeWithMpi(comm);
  }
#endif

#ifdef BUILD_CYLON_UCX
  if (COMM_ARG == "ucx") {
    LOG(INFO) << "Using UCX with MPI";
    return net::UCXConfig::Make(comm);
  }
#endif
  return nullptr;
}

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

    auto config = MakeConfig(new_comm);
    REQUIRE(config);
    std::shared_ptr<cylon::CylonContext> ctx;
    REQUIRE(cylon::CylonContext::InitDistributed(config, &ctx).is_ok());

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

    config = MakeConfig(new_comm);
    REQUIRE(config);
    REQUIRE(cylon::CylonContext::InitDistributed(config, &ctx).is_ok());

    REQUIRE(l_rank == ctx->GetRank());
    REQUIRE(ctx->GetWorldSize() == 2);

    std::string path1 = "../data/input/csv1_" + std::to_string(l_rank) + ".csv";
    std::string path2 = "../data/input/csv2_" + std::to_string(l_rank) + ".csv";
    std::string out_path =
        "../data/output/join_inner_2_" + std::to_string(l_rank) + ".csv";

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

int main(int argc, char *argv[]) {
  Catch::Session session;

  auto cli = session.cli() | Catch::clara::Opt(COMM_ARG, "mpi|gloo-mpi|ucx")["--comm"]("comm args");

  // Now pass the new composite back to Catch2 so it uses that
  session.cli(cli);

  // Let Catch2 (using Clara) parse the command line
  int returnCode = session.applyCommandLine(argc, argv);
  if (returnCode != 0) // Indicates a command line error
    return returnCode;

  LOG(INFO) << "comm args: " << COMM_ARG;
  return session.run();
}

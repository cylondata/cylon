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

#ifndef __CYLON_TEST_HEADER_
#define __CYLON_TEST_HEADER_

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include <mpi.h>
#include <iostream>
#include <glog/logging.h>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/table.hpp>
#include <chrono>
#include <cylon/net/mpi/mpi_communicator.hpp>

#ifdef BUILD_CYLON_GLOO
#include <cylon/net/gloo/gloo_communicator.hpp>
#endif // BUILD_CYLON_GLOO

#include "test_utils.hpp"
#include "test_macros.hpp"
#include "test_arrow_utils.hpp"

std::shared_ptr<cylon::CylonContext> ctx = nullptr;
int RANK = 0;
int WORLD_SZ = 0;

/**
 * --comm
 *  (0) no args --> defaults to mpi
 *  (1) mpi
 *  (2) gloo-mpi
 *  (3) gloo-standalone <rank> <world_size> TODO
 */
int main(int argc, char *argv[]) {
  Catch::Session session; // There must be exactly one instance

  std::string comm_args = "mpi";

  auto cli = session.cli() | Catch::clara::Opt(comm_args, "mpi|gloo-mpi")["--comm"]("comm args");

  // Now pass the new composite back to Catch2 so it uses that
  session.cli(cli);

  // Let Catch2 (using Clara) parse the command line
  int returnCode = session.applyCommandLine(argc, argv);
  if (returnCode != 0) // Indicates a command line error
    return returnCode;

  LOG(INFO) << "comm args: " << comm_args;

  // global setup...
  std::shared_ptr<cylon::net::CommConfig> config;
  if (comm_args == "mpi") {
    LOG(INFO) << "Using MPI";
    config = cylon::net::MPIConfig::Make();
  } else if (comm_args == "gloo-mpi") {
#ifdef BUILD_CYLON_GLOO
    LOG(INFO) << "Using Gloo with MPI";
    config = cylon::net::GlooConfig::MakeWithMpi();
#else
    LOG(ERROR) << "gloo-mpi passed for tests, but tests are not built with gloo";
    return 1;
#endif
  } else {
    LOG(ERROR) << "unsupported comm " << argv[1];
    return 1;
  }

  auto st = cylon::CylonContext::InitDistributed(config, &ctx);
  if (!st.is_ok()) {
    LOG(ERROR) << "ctx init failed: " << st.get_msg();
    return st.get_code();
  }

  RANK = ctx->GetRank();
  WORLD_SZ = ctx->GetWorldSize();

  LOG(INFO) << "wz: " << WORLD_SZ << " rank: " << RANK << std::endl;
  int result = session.run();

  // global clean-up...
  ctx->Finalize();
  return result;
}

// Other common stuff goes here ...

#endif

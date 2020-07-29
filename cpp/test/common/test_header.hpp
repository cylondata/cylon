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
#include <catch.hpp>
#include <mpi.h>
#include <iostream>
#include <glog/logging.h>
#include <ctx/cylon_context.hpp>
#include <table.hpp>
#include <chrono>
#include <net/mpi/mpi_communicator.hpp>

cylon::CylonContext *ctx = NULL;
int RANK = 0;
int WORLD_SZ = 0;

using namespace cylon;

int main(int argc, char *argv[]) {
  // global setup...
  auto mpi_config = new cylon::net::MPIConfig();
  ctx = cylon::CylonContext::InitDistributed(mpi_config);
  RANK = ctx->GetRank();
  WORLD_SZ = ctx->GetWorldSize();

  int result = Catch::Session().run(argc, argv);

  // global clean-up...
  ctx->Finalize();
  return result;
}

// Other common stuff goes here ...

#endif

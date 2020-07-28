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

using namespace cylon;

Status Table::FromCSV(cylon::CylonContext *context,
               const vector<std::string> &paths,
               const std::vector<std::shared_ptr<Table> *> &tableOuts,
               const io::config::CSVReadOptions &options) {
  // modify paths to reflect the rank
  // hypothetical...
  int rank = context->GetRank();
  // modify paths to include rank depending upon the position in the vector
  //"../data/join/table0.csv", --> "../data/join/0/table0.csv"
  // "../data/join//table1.csv" --> "../data/join/0/table1.csv"

  // read ...
  return Status::OK();
}

int main(int argc, char *argv[]) {
  // global setup...

  auto mpi_config = new cylon::net::MPIConfig();
  ctx = cylon::CylonContext::InitDistributed(mpi_config);

  int result = Catch::Session().run(argc, argv);

  // global clean-up...
  ctx->Finalize();
  return result;
}

// Other common stuff goes here ...

#endif

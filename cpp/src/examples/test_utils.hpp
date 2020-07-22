//
// Created by nira on 7/21/20.
//

#ifndef CYLON_CPP_SRC_EXAMPLES_TEST_UTILS_HPP_
#define CYLON_CPP_SRC_EXAMPLES_TEST_UTILS_HPP_

#include <glog/logging.h>
#include <net/mpi/mpi_communicator.hpp>
#include <ctx/cylon_context.hpp>
#include <table.hpp>
#include <chrono>

namespace cylon {
namespace test {
static int Verify(CylonContext *ctx, std::shared_ptr<Table> &result, std::shared_ptr<Table> &expected_result) {
  Status status;
  std::shared_ptr<Table> verification;

  LOG(INFO) << "starting verification...";
  LOG(INFO) << "expected:" << expected_result->Rows() << " found:" << result->Rows();
  status = result->Subtract(expected_result, verification);

  int rank = ctx->GetRank();

  if (!status.is_ok() || verification->Rows()) {
    LOG(ERROR) << "verification FAIL! Rank:" << rank << " status:" << status.get_msg() << " expected:"
               << expected_result->Rows() << " found:" << result->Rows();
    ctx->Finalize();
    return 1;
  } else {
    LOG(INFO) << "verification SUCCESS!";
    ctx->Finalize();
    return 0;
  }
}
}
}

#endif //CYLON_CPP_SRC_EXAMPLES_TEST_UTILS_HPP_

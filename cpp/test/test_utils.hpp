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

#ifndef CYLON_CPP_SRC_EXAMPLES_TEST_UTILS_HPP_
#define CYLON_CPP_SRC_EXAMPLES_TEST_UTILS_HPP_

#include <glog/logging.h>
#include <net/mpi/mpi_communicator.hpp>
#include <ctx/cylon_context.hpp>
#include <table.hpp>
#include <chrono>

// this is a toggle to generate test files. Set execute to 0 then, it will generate the expected
// output files
#define EXECUTE 1

namespace cylon {
namespace test {
static int Verify(CylonContext *ctx, const std::shared_ptr<Table> &result,
                  const std::shared_ptr<Table> &expected_result) {
  Status status;
  std::shared_ptr<Table> verification;

  LOG(INFO) << "starting verification...";

  LOG(INFO) << "expected:" << expected_result->Rows() << " found:" << result->Rows();

  if (!(status = result->Subtract(expected_result, verification)).is_ok()) {
    LOG(ERROR) << "subtract FAIL! " << status.get_msg();
    return 1;
  } else if (verification->Rows()) {
    LOG(ERROR) << "verification FAIL! Rank:" << ctx->GetRank() << " status:" << status.get_msg()
               << " expected:" << expected_result->Rows() << " found:" << result->Rows() << " "
               << verification->Rows();
    return 1;
  } else {
    LOG(INFO) << "verification SUCCESS!";
    return 0;
  }
}

int TestSetOperation(Status(Table::*fun_ptr)(const std::shared_ptr<Table> &right,
                                             std::shared_ptr<Table> &out),
                     CylonContext *ctx,
                     const std::string &path1,
                     const std::string &path2,
                     const std::string &out_path) {
  std::shared_ptr<cylon::Table> table1, table2, result_expected, result;

  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false)
      .WithColumnTypes({
                           {"0", std::make_shared<DataType>(Type::INT64)},
                           {"1", std::make_shared<DataType>(Type::DOUBLE)},
                       });

  Status status;
  auto start_start = std::chrono::steady_clock::now();

  status = cylon::Table::FromCSV(ctx,
#if EXECUTE
                                 std::vector<std::string>{path1, path2, out_path},
                                 std::vector<std::shared_ptr<Table> *>{&table1, &table2,
                                                                       &result_expected},
#else
      std::vector<std::string>{path1, path2},
      std::vector<std::shared_ptr<Table> *>{&table1, &table2},
#endif
                                 read_options);

  if (!status.is_ok()) {
    LOG(INFO) << "Table reading failed " << status.get_msg();
    ctx->Finalize();
    return 1;
  }

  auto read_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "Read tables in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(read_end_time - start_start)
                .count()
            << "[ms]";

  if (!(status = ((*table1).*fun_ptr)(table2, result)).is_ok()) {
    LOG(INFO) << "Table op failed ";
    ctx->Finalize();
    return 1;
  }
  auto op_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "First table had : " << table1->Rows() << " and Second table had : "
            << table2->Rows() << ", result has : " << result->Rows();
  LOG(INFO) << "operation done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(op_end_time - read_end_time)
                .count()
            << "[ms]";

#if EXECUTE
  return test::Verify(ctx, result, result_expected);
#else
  auto write_options = io::config::CSVWriteOptions().ColumnNames(result->ColumnNames());
  result->WriteCSV(out_path, write_options);
  return 0;
#endif
}

int TestJoinOperation(const cylon::join::config::JoinConfig &join_config,
                      CylonContext *ctx,
                      const std::string &path1,
                      const std::string &path2,
                      const std::string &out_path) {
  Status status;
  std::shared_ptr<cylon::Table> table1, table2, joined_expected, joined, verification;

  auto start_start = std::chrono::steady_clock::now();

  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false);
  status = cylon::Table::FromCSV(ctx,
#if EXECUTE
                                 std::vector<std::string>{path1, path2, out_path},
                                 std::vector<std::shared_ptr<Table> *>{&table1, &table2,
                                                                       &joined_expected},
#else
      std::vector<std::string>{path1, path2},
      std::vector<std::shared_ptr<Table> *>{&table1, &table2},
#endif
                                 read_options);

  if (!status.is_ok()) {
    LOG(INFO) << "Table reading failed " << status.get_msg();
    return 1;
  }

  auto read_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "Read tables in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(read_end_time - start_start)
                .count()
            << "[ms]";

  status = table1->DistributedJoin(table2, join_config, &joined);
  if (!status.is_ok()) {
    LOG(INFO) << "Table join failed ";
    return 1;
  }
  auto join_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "First table had : " << table1->Rows() << " and Second table had : "
            << table2->Rows() << ", Joined has : " << joined->Rows();
  LOG(INFO) << "Join done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(join_end_time - read_end_time)
                .count()
            << "[ms]";

#if EXECUTE
  if (test::Verify(ctx, joined, joined_expected)) {
    LOG(ERROR) << "join failed!";
    return 1;
  }
#else
  auto write_options = io::config::CSVWriteOptions().ColumnNames(joined->ColumnNames());
  joined->WriteCSV(out_path, write_options);
#endif
  return 0;
}

}
}

#endif //CYLON_CPP_SRC_EXAMPLES_TEST_UTILS_HPP_

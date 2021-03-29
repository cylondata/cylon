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

#define LOG_AND_RETURN_INT_IF_FAILED(status) \
  if (!status.is_ok()) { \
    LOG(ERROR) << status.get_msg() ; \
    return status.get_code(); \
  };


// this is a toggle to generate test files. Set execute to 0 then, it will generate the expected
// output files
#define EXECUTE 1

namespace cylon {
namespace test {
static int Verify(std::shared_ptr<cylon::CylonContext> &ctx, std::shared_ptr<Table> &result,
                  std::shared_ptr<Table> &expected_result) {
  Status status;
  std::shared_ptr<Table> verification;

  LOG(INFO) << "starting verification...";

  if (expected_result->Rows() != result->Rows()) {
    LOG(ERROR) << "expected:" << expected_result->Rows() << " found:" << result->Rows();
    return 1;
  } else if (!(status = cylon::Subtract(result, expected_result, verification)).is_ok()) {
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

typedef Status(*fun_ptr)(std::shared_ptr<Table> &,
                         std::shared_ptr<Table> &,
                         std::shared_ptr<Table> &);

int TestSetOperation(fun_ptr fn,
                     std::shared_ptr<cylon::CylonContext> &ctx,
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

  status = cylon::FromCSV(ctx,
#if EXECUTE
                          std::vector<std::string>{path1, path2, out_path},
                          std::vector<std::shared_ptr<Table> *>{&table1, &table2,
                                                                &result_expected},
#else
      std::vector<std::string>{path1, path2},
      std::vector<std::shared_ptr<Table> *>{&table1, &table2},
#endif
                          read_options);
  LOG_AND_RETURN_INT_IF_FAILED(status)

  auto read_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "Read tables in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(read_end_time - start_start)
                .count()
            << "[ms]";
  status = fn(table1, table2, result);
  LOG_AND_RETURN_INT_IF_FAILED(status)

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
  LOG_AND_RETURN_INT_IF_FAILED(WriteCSV(result, out_path, write_options))
  return 0;
#endif
}

int TestJoinOperation(const cylon::join::config::JoinConfig &join_config,
                      std::shared_ptr<cylon::CylonContext> &ctx,
                      const std::string &path1,
                      const std::string &path2,
                      const std::string &out_path) {
  Status status;
  std::shared_ptr<cylon::Table> table1, table2, joined_expected, joined, verification;

  auto start_start = std::chrono::steady_clock::now();

  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false);
  status = cylon::FromCSV(ctx,
#if EXECUTE
                          std::vector<std::string>{path1, path2, out_path},
                          std::vector<std::shared_ptr<Table> *>{&table1, &table2,
                                                                &joined_expected},
#else
      std::vector<std::string>{path1, path2},
      std::vector<std::shared_ptr<Table> *>{&table1, &table2},
#endif
                          read_options);
  LOG_AND_RETURN_INT_IF_FAILED(status)

  auto read_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "Read tables in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(read_end_time - start_start)
                .count()
            << "[ms]";

  status = cylon::DistributedJoin(table1, table2, join_config, joined);
  LOG_AND_RETURN_INT_IF_FAILED(status)

  auto join_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "First table had : " << table1->Rows() << " and Second table had : "
            << table2->Rows() << ", Joined has : " << joined->Rows();
  LOG(INFO) << "Join done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(join_end_time - read_end_time)
                .count()
            << "[ms]";

#if EXECUTE
  return test::Verify(ctx, joined, joined_expected);
#else
  auto write_options = io::config::CSVWriteOptions().ColumnNames(joined->ColumnNames());
  LOG_AND_RETURN_INT_IF_FAILED(cylon::WriteCSV(joined, out_path, write_options));
  return 0;
#endif
}

cylon::Status CreateTable(std::shared_ptr<cylon::CylonContext> &ctx, int rows, std::shared_ptr<cylon::Table> &output) {
  std::shared_ptr<std::vector<int32_t>> col0 = std::make_shared<std::vector<int32_t >>();
  std::shared_ptr<std::vector<double_t>> col1 = std::make_shared<std::vector<double_t >>();

  for (int i = 0; i < rows; i++) {
    col0->push_back(i);
    col1->push_back((double_t) i + 10.0);
  }

  auto c0 = cylon::VectorColumn<int32_t>::Make("col0", cylon::Int32(), col0);
  auto c1 = cylon::VectorColumn<double>::Make("col1", cylon::Double(), col1);

  return cylon::Table::FromColumns(ctx, {c0, c1}, output);
}

#ifdef BUILD_CYLON_PARQUET
int TestParquetJoinOperation(const cylon::join::config::JoinConfig &join_config,
                      std::shared_ptr<cylon::CylonContext> &ctx,
                      const std::string &path1,
                      const std::string &path2,
                      const std::string &out_path) {
  Status status;
  std::shared_ptr<cylon::Table> table1, table2, joined_expected, joined, verification;

  auto start_start = std::chrono::steady_clock::now();

  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false);
  status = cylon::FromParquet(ctx,
#if EXECUTE
                          std::vector<std::string>{path1, path2, out_path},
                          std::vector<std::shared_ptr<Table> *>{&table1, &table2,
                                                                &joined_expected}
#else
      std::vector<std::string>{path1, path2},
      std::vector<std::shared_ptr<Table> *>{&table1, &table2}
#endif
                          );
  LOG_AND_RETURN_INT_IF_FAILED(status)

  auto read_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "Read tables in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(read_end_time - start_start)
                .count()
            << "[ms]";

  status = cylon::DistributedJoin(table1, table2, join_config, joined);
  LOG_AND_RETURN_INT_IF_FAILED(status)

  auto join_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "First table had : " << table1->Rows() << " and Second table had : "
            << table2->Rows() << ", Joined has : " << joined->Rows();
  LOG(INFO) << "Join done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(join_end_time - read_end_time)
                .count()
            << "[ms]";

#if EXECUTE
  return test::Verify(ctx, joined, joined_expected);
#else
  auto parquetOptions = cylon::io::config::ParquetOptions();
  LOG_AND_RETURN_INT_IF_FAILED(cylon::WriteParquet(joined, ctx, out_path, parquetOptions))
  return 0;
#endif
}
#endif

template<typename T>
std::shared_ptr<arrow::Array> VectorToArrowArray(const std::vector<T> &v) {
  const auto &buf = arrow::Buffer::Wrap(v);
  const auto
      &data = arrow::ArrayData::Make(arrow::TypeTraits<typename arrow::CTypeTraits<T>::ArrowType>::type_singleton(),
                                     v.size(),
                                     {nullptr, buf});

  return arrow::MakeArray(data);
}

}
}

#endif //CYLON_CPP_SRC_EXAMPLES_TEST_UTILS_HPP_

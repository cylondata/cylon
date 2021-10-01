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
#include <chrono>

#include <cylon/table.hpp>
#include "test_macros.hpp"

// this is a toggle to generate test files. Set execute to 0 then, it will generate the expected
// output files
#define EXECUTE 1

namespace cylon {
namespace test {

typedef Status(*fun_ptr)(std::shared_ptr<Table> &,
                         std::shared_ptr<Table> &,
                         std::shared_ptr<Table> &);

void TestSetOperation(fun_ptr fn,
                      std::shared_ptr<CylonContext> &ctx,
                      const std::string &path1,
                      const std::string &path2,
                      const std::string &out_path) {
  std::shared_ptr<Table> table1, table2, result_expected, result;

  auto read_options = io::config::CSVReadOptions().UseThreads(false)
      .WithColumnTypes({
                           {"0", std::make_shared<DataType>(Type::INT64)},
                           {"1", std::make_shared<DataType>(Type::DOUBLE)},
                       });

#if EXECUTE
  CHECK_CYLON_STATUS(FromCSV(ctx, std::vector<std::string>{path1, path2, out_path},
                             std::vector<std::shared_ptr<Table> *>{&table1, &table2, &result_expected},
                             read_options));
#else
  CHECK_CYLON_STATUS(FromCSV(ctx, std::vector<std::string>{path1, path2},
                             std::vector<std::shared_ptr<Table> *>{&table1, &table2},
                             read_options));
#endif

  CHECK_CYLON_STATUS(fn(table1, table2, result));

#if EXECUTE
  VERIFY_TABLES_EQUAL_UNORDERED(result_expected, result);
#else
  auto write_options = io::config::CSVWriteOptions().ColumnNames(result->ColumnNames());
  CHECK_CYLON_STATUS(WriteCSV(result, out_path, write_options));
#endif
}

void TestJoinOperation(const join::config::JoinConfig &join_config,
                       const std::shared_ptr<CylonContext> &ctx,
                       const std::string &path1,
                       const std::string &path2,
                       const std::string &out_path) {
  std::shared_ptr<Table> table1, table2, expected, joined, verification;

  auto read_options = io::config::CSVReadOptions().UseThreads(false);

#if EXECUTE
  CHECK_CYLON_STATUS(FromCSV(ctx, std::vector<std::string>{path1, path2, out_path},
                             std::vector<std::shared_ptr<Table> *>{&table1, &table2, &expected},
                             read_options));
#else
  CHECK_CYLON_STATUS(FromCSV(ctx, std::vector<std::string>{path1, path2},
                             std::vector<std::shared_ptr<Table> *>{&table1, &table2},
                             read_options));
#endif

  CHECK_CYLON_STATUS(DistributedJoin(table1, table2, join_config, joined));

#if EXECUTE
  VERIFY_TABLES_EQUAL_UNORDERED(expected, joined);
#else
  auto write_options = io::config::CSVWriteOptions().ColumnNames(joined->ColumnNames());
  LOG_AND_RETURN_INT_IF_FAILED(WriteCSV(joined, out_path, write_options));
#endif
}

Status CreateTable(const std::shared_ptr<CylonContext> &ctx, int rows, std::shared_ptr<Table> &output) {
  std::vector<int32_t> col0;
  std::vector<double_t> col1;

  for (int i = 0; i < rows; i++) {
    col0.push_back(i);
    col1.push_back((double_t) i + 10.0);
  }

  std::shared_ptr<Column> c0, c1;
  RETURN_CYLON_STATUS_IF_FAILED(Column::FromVector(ctx, "col0", Int32(), col0, c0));
  RETURN_CYLON_STATUS_IF_FAILED(Column::FromVector(ctx, "col1", Double(), col1, c1));

  return Table::FromColumns(ctx, {std::move(c0), std::move(c1)}, output);
}

#ifdef BUILD_CYLON_PARQUET
void TestParquetJoinOperation(const join::config::JoinConfig &join_config,
                              std::shared_ptr<CylonContext> &ctx,
                              const std::string &path1,
                              const std::string &path2,
                              const std::string &out_path) {
  std::shared_ptr<Table> table1, table2, expected, joined, verification;

  auto read_options = io::config::ParquetOptions().ConcurrentFileReads(false);
#if EXECUTE
  CHECK_CYLON_STATUS(FromParquet(ctx, std::vector<std::string>{path1, path2, out_path},
                                 std::vector<std::shared_ptr<Table> *>{&table1, &table2, &expected},
                                 read_options));
#else
  CHECK_CYLON_STATUS(FromParquet(ctx, std::vector<std::string>{path1, path2},
                             std::vector<std::shared_ptr<Table> *>{&table1, &table2},
                             read_options));
#endif

  CHECK_CYLON_STATUS(DistributedJoin(table1, table2, join_config, joined));

#if EXECUTE
  VERIFY_TABLES_EQUAL_UNORDERED(expected, joined);
#else
  auto parquetOptions = io::config::ParquetOptions();
  LOG_AND_RETURN_INT_IF_FAILED(WriteParquet(joined, ctx, out_path, parquetOptions))
#endif
}
#endif

template<typename T>
std::shared_ptr<arrow::Array> VectorToArrowArray(const std::vector<T> &v) {
  const auto &buf = arrow::Buffer::Wrap(v);
  const auto &type = arrow::TypeTraits<typename arrow::CTypeTraits<T>::ArrowType>::type_singleton();
  const auto &data = arrow::ArrayData::Make(type, v.size(), {nullptr, buf});
  return arrow::MakeArray(data);
}

}
}

#endif //CYLON_CPP_SRC_EXAMPLES_TEST_UTILS_HPP_

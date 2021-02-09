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

#include "indexing/index_utils.hpp"
#include "indexing/indexer.hpp"

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

  LOG(INFO) << "expected:" << expected_result->Rows() << " found:" << result->Rows();

  if (!(status = cylon::Subtract(result, expected_result, verification)).is_ok()) {
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
  status = fn(table1, table2, result);
  if (!status.is_ok()) {
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
  WriteCSV(result, out_path, write_options);
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

  if (!status.is_ok()) {
    LOG(INFO) << "Table reading failed " << status.get_msg();
    return 1;
  }

  auto read_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "Read tables in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(read_end_time - start_start)
                .count()
            << "[ms]";

  status = cylon::DistributedJoin(table1, table2, join_config, joined);
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
  cylon::WriteCSV(joined, out_path, write_options);
#endif
  return 0;
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

  if (!status.is_ok()) {
    LOG(INFO) << "Table reading failed " << status.get_msg();
    return 1;
  }

  auto read_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "Read tables in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(read_end_time - start_start)
                .count()
            << "[ms]";

  status = cylon::DistributedJoin(table1, table2, join_config, joined);
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
  auto parquetOptions = cylon::io::config::ParquetOptions().ChunkSize(5);
  cylon::WriteParquet(joined, ctx, out_path, parquetOptions);
#endif
  return 0;
}
#endif

int TestIndexBuildOperation(std::string &input_file_path, cylon::IndexingSchema indexing_schema) {
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input1, output;
  std::shared_ptr<cylon::BaseIndex> index;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << input_file_path << std::endl;
  status = cylon::FromCSV(ctx, input_file_path, input1, read_options);

  const int index_column = 0;

  status = cylon::IndexUtil::BuildIndex(indexing_schema, input1, index_column, index);

  if (!status.is_ok()) {
    return 1;
  }

  bool valid_index;

  valid_index = input1->Rows() == index->GetIndexArray()->length();

  if (!valid_index) {
    return 1;
  };

  return 0;
}

static int set_data_for_indexing_test(std::string &input_file_path,
                                      cylon::IndexingSchema indexing_schema,
                                      std::string &output_file_path,
                                      std::shared_ptr<cylon::Table> &input,
                                      std::shared_ptr<cylon::Table> &expected_output,
                                      std::shared_ptr<cylon::BaseIndex> &index,
                                      int id
) {

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;
  output_file_path = output_file_path + std::to_string(id) + ".csv";
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << input_file_path << std::endl;
  status = cylon::FromCSV(ctx, input_file_path, input, read_options);

  if(!status.is_ok()) {
    return 1;
  }

  status = cylon::FromCSV(ctx, output_file_path, expected_output, read_options);

  if(!status.is_ok()) {
    return 1;
  }

  const int index_column = 0;
  bool drop_index = true;

  status = cylon::IndexUtil::BuildIndex(indexing_schema, input, index_column, index);

  if (!status.is_ok()) {
    return 1;
  }

  bool valid_index = false;
  if (indexing_schema == cylon::IndexingSchema::Range) {
    valid_index = input->Rows() == index->GetSize();
  } else {
    valid_index = input->Rows() == index->GetIndexArray()->length();
  }

  if (!valid_index) {
    return 1;
  };

  status = input->Set_Index(index, drop_index);

  if (!status.is_ok()) {
    return 1;
  }

  return 0;
}


int TestIndexLocOperation1(std::string &input_file_path,
                           cylon::IndexingSchema indexing_schema,
                           std::string &output_file_path) {
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;
  std::shared_ptr<cylon::Table> input, expected_output, result;
  std::shared_ptr<BaseIndex> index;
  int res = set_data_for_indexing_test(input_file_path, indexing_schema, output_file_path, input, expected_output, index, 1);

  if(res != 0) {
    return res;
  }

  long start_index = 0;
  long end_index = 5;
  int column = 0;

  std::shared_ptr<cylon::BaseIndexer> indexer = std::make_shared<cylon::LocIndexer>(indexing_schema);

  status = indexer->loc(&start_index, &end_index, column, input, result);

  if (!status.is_ok()) {
    return -1;
  }

  //auto write_options = io::config::CSVWriteOptions();
  //cylon::WriteCSV(result, "/tmp/indexing_result_check" + std::to_string(indexing_schema) + ".csv", write_options);

  if (test::Verify(ctx, result, expected_output)) {
    LOG(ERROR) << "Loc 1 failed!";
    return 1;
  }

  return 0;
}



int TestIndexLocOperation2(std::string &input_file_path,
                           cylon::IndexingSchema indexing_schema,
                           std::string &output_file_path) {
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, expected_output, result;
  std::shared_ptr<cylon::BaseIndex> index;
  int res = set_data_for_indexing_test(input_file_path, indexing_schema, output_file_path, input, expected_output, index, 2);

  if(res != 0) {
    return res;
  }

  long start_index = 0;
  long end_index = 5;
  int start_column_index = 0;
  int end_column_index = 1;

  std::shared_ptr<cylon::BaseIndexer> indexer = std::make_shared<cylon::LocIndexer>(indexing_schema);

  status = indexer->loc(&start_index, &end_index, start_column_index, end_column_index, input, result);

  if (!status.is_ok()) {
    return -1;
  }

  if (test::Verify(ctx, result, expected_output)) {
    LOG(ERROR) << "Loc 2 failed!";
    return 1;
  }

  return 0;
}


int TestIndexLocOperation3(std::string &input_file_path,
                           cylon::IndexingSchema indexing_schema,
                           std::string &output_file_path) {
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, expected_output, result;
  std::shared_ptr<cylon::BaseIndex> index;
  int res = set_data_for_indexing_test(input_file_path, indexing_schema, output_file_path, input, expected_output, index, 3);

  if(res != 0) {
    return res;
  }

  long start_index = 0;
  long end_index = 5;
  std::vector<int> cols = {0,2};

  std::shared_ptr<cylon::BaseIndexer> indexer = std::make_shared<cylon::LocIndexer>(indexing_schema);

  status = indexer->loc(&start_index, &end_index, cols, input, result);

  if (!status.is_ok()) {
    return -1;
  }

  if (test::Verify(ctx, result, expected_output)) {
    LOG(ERROR) << "Loc 3 failed!";
    return 1;
  }

  return 0;
}

int TestIndexLocOperation4(std::string &input_file_path,
                           cylon::IndexingSchema indexing_schema,
                           std::string &output_file_path) {
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, expected_output, result;
  std::shared_ptr<cylon::BaseIndex> index;
  int res = set_data_for_indexing_test(input_file_path, indexing_schema, output_file_path, input, expected_output, index, 4);

  if(res != 0) {
    return res;
  }

  long start_index = 10;
  int column = 1;

  std::shared_ptr<cylon::BaseIndexer> indexer = std::make_shared<cylon::LocIndexer>(indexing_schema);

  status = indexer->loc(&start_index, column, input, result);

  if (!status.is_ok()) {
    return -1;
  }

  if (test::Verify(ctx, result, expected_output)) {
    LOG(ERROR) << "Loc 4 failed!";
    return 1;
  }

  return 0;
}


int TestIndexLocOperation5(std::string &input_file_path,
                           cylon::IndexingSchema indexing_schema,
                           std::string &output_file_path) {
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, expected_output, result;
  std::shared_ptr<cylon::BaseIndex> index;
  int res = set_data_for_indexing_test(input_file_path, indexing_schema, output_file_path, input, expected_output, index, 5);

  if(res != 0) {
    return res;
  }

  typedef std::vector<void *> vector_void_star;

  vector_void_star output_items;

  std::vector<long> start_indices = {4, 10};
  int start_column = 1;
  int end_column = 2;

  for (size_t tx = 0; tx < start_indices.size(); tx++) {
    long *val = new long(start_indices.at(tx));
    output_items.push_back(static_cast<void *>(val));
  }

  std::shared_ptr<cylon::BaseIndexer> indexer = std::make_shared<cylon::LocIndexer>(indexing_schema);

  status = indexer->loc(output_items, start_column, end_column, input, result);

  if (!status.is_ok()) {
    return -1;
  }

  if (test::Verify(ctx, result, expected_output)) {
    LOG(ERROR) << "Loc 5 failed!";
    return 1;
  }

  return 0;
}

int TestIndexLocOperation6(std::string &input_file_path,
                           cylon::IndexingSchema indexing_schema,
                           std::string &output_file_path) {
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, expected_output, result;
  std::shared_ptr<cylon::BaseIndex> index;
  int res = set_data_for_indexing_test(input_file_path, indexing_schema, output_file_path, input, expected_output, index, 6);

  if(res != 0) {
    return res;
  }

  typedef std::vector<void *> vector_void_star;

  vector_void_star output_items;

  std::vector<long> start_indices = {4, 10};
  std::vector<int> columns = {0, 2};

  for (size_t tx = 0; tx < start_indices.size(); tx++) {
    long *val = new long(start_indices.at(tx));
    output_items.push_back(static_cast<void *>(val));
  }

  std::shared_ptr<cylon::BaseIndexer> indexer = std::make_shared<cylon::LocIndexer>(indexing_schema);

  status = indexer->loc(output_items, columns, input, result);

  if (!status.is_ok()) {
    return -1;
  }

  if (test::Verify(ctx, result, expected_output)) {
    LOG(ERROR) << "Loc 6 failed!";
    return 1;
  }

  return 0;
}

int TestIndexLocOperation7(std::string &input_file_path,
                           cylon::IndexingSchema indexing_schema,
                           std::string &output_file_path) {
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, expected_output, result;
  std::shared_ptr<cylon::BaseIndex> index;
  int res = set_data_for_indexing_test(input_file_path, indexing_schema, output_file_path, input, expected_output, index, 7);

  if(res != 0) {
    return res;
  }


  long start_index = 4;
  std::vector<int> columns = {0, 1};

  std::shared_ptr<cylon::BaseIndexer> indexer = std::make_shared<cylon::LocIndexer>(indexing_schema);

  status = indexer->loc(&start_index, columns, input, result);

  if (!status.is_ok()) {
    return -1;
  }

  if (test::Verify(ctx, result, expected_output)) {
    LOG(ERROR) << "Loc 7 failed!";
    return 1;
  }

  return 0;
}

int TestIndexLocOperation8(std::string &input_file_path,
                           cylon::IndexingSchema indexing_schema,
                           std::string &output_file_path) {
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, expected_output, result;
  std::shared_ptr<cylon::BaseIndex> index;
  int res = set_data_for_indexing_test(input_file_path, indexing_schema, output_file_path, input, expected_output, index, 8);

  if(res != 0) {
    return res;
  }


  long start_index = 4;
  int start_column = 1;
  int end_column = 2;

  std::shared_ptr<cylon::BaseIndexer> indexer = std::make_shared<cylon::LocIndexer>(indexing_schema);

  status = indexer->loc(&start_index, start_column, end_column, input, result);

  if (!status.is_ok()) {
    return -1;
  }

  if (test::Verify(ctx, result, expected_output)) {
    LOG(ERROR) << "Loc 8 failed!";
    return 1;
  }

  return 0;
}

int TestIndexLocOperation9(std::string &input_file_path,
                           cylon::IndexingSchema indexing_schema,
                           std::string &output_file_path) {
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, expected_output, result;
  std::shared_ptr<cylon::BaseIndex> index;
  int res = set_data_for_indexing_test(input_file_path, indexing_schema, output_file_path, input, expected_output, index, 9);

  if(res != 0) {
    return res;
  }


  typedef std::vector<void *> vector_void_star;

  vector_void_star output_items;

  std::vector<long> start_indices = {4, 10};
  int column = 0;

  for (size_t tx = 0; tx < start_indices.size(); tx++) {
    long *val = new long(start_indices.at(tx));
    output_items.push_back(static_cast<void *>(val));
  }

  std::shared_ptr<cylon::BaseIndexer> indexer = std::make_shared<cylon::LocIndexer>(indexing_schema);

  status = indexer->loc(output_items, column, input, result);

  if (!status.is_ok()) {
    return -1;
  }

  if (test::Verify(ctx, result, expected_output)) {
    LOG(ERROR) << "Loc 9 failed!";
    return 1;
  }

  return 0;
}

}
}

#endif //CYLON_CPP_SRC_EXAMPLES_TEST_UTILS_HPP_

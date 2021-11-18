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

#include <glog/logging.h>
#include <chrono>
#include <map>

#include <cylon/net/mpi/mpi_communicator.hpp>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/table.hpp>
#include <cylon/ctx/arrow_memory_pool_utils.hpp>
#include <cylon/indexing/index_utils.hpp>
#include <cylon/indexing/indexer.hpp>
#include <cylon/indexing/index.hpp>
#include "test_utils.hpp"
#include "test_macros.hpp"
#include "test_arrow_utils.hpp"

namespace cylon {
namespace test {

void TestIndexBuildOperation(const std::shared_ptr<CylonContext> &ctx,
                             std::string &input_file_path,
                             cylon::IndexingType indexing_type) {

  std::shared_ptr<cylon::Table> input1, output;
  std::shared_ptr<cylon::BaseArrowIndex> index;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  INFO("Reading File [" << ctx->GetRank() << "] : " << input_file_path);
  CHECK_CYLON_STATUS(FromCSV(ctx, input_file_path, input1, read_options));

  int index_column = 0;

  CHECK_CYLON_STATUS(IndexUtil::BuildArrowIndex(indexing_type, input1, index_column, index));

  REQUIRE(input1->Rows() == index->GetIndexArray()->length());
}

void set_data_for_arrow_indexing_test(const std::shared_ptr<CylonContext> &ctx,
                                      std::string &input_file_path,
                                      cylon::IndexingType indexing_type,
                                      std::string &output_file_path,
                                      std::shared_ptr<Table> &input,
                                      std::shared_ptr<Table> &expected_output,
                                      int id) {
  output_file_path = output_file_path + std::to_string(id) + ".csv";
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  INFO("Reading File [" << ctx->GetRank() << "] : " << input_file_path << " " << output_file_path);
  CHECK_CYLON_STATUS(FromCSV(ctx, input_file_path, input, read_options));
  CHECK_CYLON_STATUS(FromCSV(ctx, output_file_path, expected_output, read_options));

  int index_column = 0;
  bool drop_index = true;

  std::shared_ptr<cylon::BaseArrowIndex> index;
  CHECK_CYLON_STATUS(IndexUtil::BuildArrowIndex(indexing_type, input, index_column, index));

  if (indexing_type == cylon::IndexingType::Range) {
    REQUIRE(input->Rows() == index->GetSize());
  } else {
    REQUIRE(input->Rows() == index->GetIndexArray()->length());
  }

  CHECK_CYLON_STATUS(input->SetArrowIndex(index, drop_index));
}

void TestIndexLocOperation1(const std::shared_ptr<CylonContext> &ctx,
                            std::string &input_file_path,
                            IndexingType indexing_type,
                            std::string &output_file_path) {
  INFO("index type: " << std::to_string(indexing_type) << " input: " << input_file_path << " expected: "
                      << output_file_path);
  std::shared_ptr<Table> input, expected_output, result;
  set_data_for_arrow_indexing_test(ctx, input_file_path, indexing_type, output_file_path, input, expected_output, 1);

  auto start_index = arrow::MakeScalar<int64_t>(0);
  auto end_index = arrow::MakeScalar<int64_t>(5);
  int column = 0;

  auto indexer = std::make_shared<cylon::ArrowLocIndexer>(indexing_type);

  CHECK_CYLON_STATUS(indexer->loc(start_index, end_index, column, input, result));
  VERIFY_TABLES_EQUAL_UNORDERED(expected_output, result);
}

void TestIndexLocOperation2(const std::shared_ptr<CylonContext> &ctx,
                            std::string &input_file_path,
                            IndexingType indexing_type,
                            std::string &output_file_path) {
  INFO("index type: " << std::to_string(indexing_type) << " input: " << input_file_path << " expected: "
                      << output_file_path);

  std::shared_ptr<Table> input, expected_output, result;
  set_data_for_arrow_indexing_test(ctx, input_file_path, indexing_type, output_file_path, input, expected_output, 2);

  auto start_index = arrow::MakeScalar<int64_t>(0);
  auto end_index = arrow::MakeScalar<int64_t>(5);
  int start_column_index = 0;
  int end_column_index = 1;

  auto indexer = std::make_shared<cylon::ArrowLocIndexer>(indexing_type);

  CHECK_CYLON_STATUS(indexer->loc(start_index, end_index, start_column_index, end_column_index, input, result));
  VERIFY_TABLES_EQUAL_UNORDERED(expected_output, result);
}

void TestIndexLocOperation3(const std::shared_ptr<CylonContext> &ctx, std::string &input_file_path,
                            IndexingType indexing_type,
                            std::string &output_file_path) {
  INFO("index type: " << std::to_string(indexing_type) << " input: " << input_file_path << " expected: "
                      << output_file_path);
  std::shared_ptr<Table> input, expected_output, result;
  set_data_for_arrow_indexing_test(ctx, input_file_path, indexing_type, output_file_path, input, expected_output, 3);

  auto start_index = arrow::MakeScalar<int64_t>(0);
  auto end_index = arrow::MakeScalar<int64_t>(5);
  std::vector<int> cols = {0, 2};

  auto indexer = std::make_shared<cylon::ArrowLocIndexer>(indexing_type);

  CHECK_CYLON_STATUS(indexer->loc(start_index, end_index, cols, input, result));
  VERIFY_TABLES_EQUAL_UNORDERED(expected_output, result);
}

void TestIndexLocOperation4(const std::shared_ptr<CylonContext> &ctx, std::string &input_file_path,
                            IndexingType indexing_type,
                            std::string &output_file_path) {
  INFO("index type: " << std::to_string(indexing_type) << " input: " << input_file_path << " expected: "
                      << output_file_path);

  std::shared_ptr<Table> input, expected_output, result;
  set_data_for_arrow_indexing_test(ctx, input_file_path, indexing_type, output_file_path, input, expected_output, 4);

  auto start_index = ArrayFromJSON(arrow::int64(), "[10]");
  int column = 1;

  auto indexer = std::make_shared<cylon::ArrowLocIndexer>(indexing_type);

  CHECK_CYLON_STATUS(indexer->loc(start_index, column, input, result));
  VERIFY_TABLES_EQUAL_UNORDERED(expected_output, result);
}

void TestIndexLocOperation5(const std::shared_ptr<CylonContext> &ctx, std::string &input_file_path,
                            IndexingType indexing_type, std::string &output_file_path) {
  INFO("index type: " << std::to_string(indexing_type) << " input: " << input_file_path << " expected: "
                      << output_file_path);

  std::shared_ptr<Table> input, expected_output, result;
  set_data_for_arrow_indexing_test(ctx, input_file_path, indexing_type, output_file_path, input, expected_output, 5);

  const auto &start_index = ArrayFromJSON(arrow::int64(), "[4, 10]");
  int start_column = 1;
  int end_column = 2;

  auto indexer = std::make_shared<cylon::ArrowLocIndexer>(indexing_type);

  CHECK_CYLON_STATUS(indexer->loc(start_index, start_column, end_column, input, result));
  VERIFY_TABLES_EQUAL_UNORDERED(expected_output, result);
}

void TestIndexLocOperation6(const std::shared_ptr<CylonContext> &ctx, std::string &input_file_path,
                            IndexingType indexing_type,
                            std::string &output_file_path) {
  INFO("index type: " << std::to_string(indexing_type) << " input: " << input_file_path << " expected: "
                      << output_file_path);

  std::shared_ptr<Table> input, expected_output, result;
  set_data_for_arrow_indexing_test(ctx, input_file_path, indexing_type, output_file_path, input, expected_output, 6);

  std::vector<int> columns = {0, 2};
  auto start_index = ArrayFromJSON(arrow::int64(), "[4, 10]");

  auto indexer = std::make_shared<cylon::ArrowLocIndexer>(indexing_type);

  CHECK_CYLON_STATUS(indexer->loc(start_index, columns, input, result));
  VERIFY_TABLES_EQUAL_UNORDERED(expected_output, result);
}

/**
 * Remove Operation 7-9 since Arrow-based indexer does support them with cases 4, 5, 6
 * **/

//int TestIndexLocOperation7(std::string &input_file_path,
//                           cylon::IndexingSchema indexing_schema,
//                           std::string &output_file_path) {
//  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
//  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
//
//  cylon::Status status;
//
//  std::shared_ptr<cylon::Table> input, expected_output, result;
//  std::shared_ptr<cylon::BaseIndex> index;
//  int res = set_data_for_indexing_test(input_file_path, indexing_schema, output_file_path, input, expected_output, index, 7);
//
//  if(res != 0) {
//    return res;
//  }
//
//
//  long start_index = 4;
//  std::vector<int> columns = {0, 1};
//
//  std::shared_ptr<cylon::BaseIndexer> indexer = std::make_shared<cylon::LocIndexer>(indexing_schema);
//
//  status = indexer->loc(&start_index, columns, input, result);
//
//  if (!status.is_ok()) {
//    return -1;
//  }
//
//  if (test::Verify(ctx, result, expected_output)) {
//    LOG(ERROR) << "Loc 7 failed!";
//    return 1;
//  }
//
//  return 0;
//}
//
//int TestIndexLocOperation8(std::string &input_file_path,
//                           cylon::IndexingSchema indexing_schema,
//                           std::string &output_file_path) {
//  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
//  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
//
//  cylon::Status status;
//
//  std::shared_ptr<cylon::Table> input, expected_output, result;
//  std::shared_ptr<cylon::BaseIndex> index;
//  int res = set_data_for_indexing_test(input_file_path, indexing_schema, output_file_path, input, expected_output, index, 8);
//
//  if(res != 0) {
//    return res;
//  }
//
//
//  long start_index = 4;
//  int start_column = 1;
//  int end_column = 2;
//
//  std::shared_ptr<cylon::BaseIndexer> indexer = std::make_shared<cylon::LocIndexer>(indexing_schema);
//
//  status = indexer->loc(&start_index, start_column, end_column, input, result);
//
//  if (!status.is_ok()) {
//    return -1;
//  }
//
//  if (test::Verify(ctx, result, expected_output)) {
//    LOG(ERROR) << "Loc 8 failed!";
//    return 1;
//  }
//
//  return 0;
//}
//
//int TestIndexLocOperation9(std::string &input_file_path,
//                           cylon::IndexingSchema indexing_schema,
//                           std::string &output_file_path) {
//  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
//  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
//
//  cylon::Status status;
//
//  std::shared_ptr<cylon::Table> input, expected_output, result;
//  std::shared_ptr<cylon::BaseIndex> index;
//  int res = set_data_for_indexing_test(input_file_path, indexing_schema, output_file_path, input, expected_output, index, 9);
//
//  if(res != 0) {
//    return res;
//  }
//
//
//  typedef std::vector<void *> vector_void_star;
//
//  vector_void_star output_items;
//
//  std::vector<long> start_indices = {4, 10};
//  int column = 0;
//
//  for (size_t tx = 0; tx < start_indices.size(); tx++) {
//    long *val = new long(start_indices.at(tx));
//    output_items.push_back(static_cast<void *>(val));
//  }
//
//  std::shared_ptr<cylon::BaseIndexer> indexer = std::make_shared<cylon::LocIndexer>(indexing_schema);
//
//  status = indexer->loc(output_items, column, input, result);
//
//  if (!status.is_ok()) {
//    return -1;
//  }
//
//  if (test::Verify(ctx, result, expected_output)) {
//    LOG(ERROR) << "Loc 9 failed!";
//    return 1;
//  }
//
//  return 0;
//}

}
}

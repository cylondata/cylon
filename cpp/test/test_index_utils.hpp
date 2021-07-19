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
#include "test_utils.hpp"

namespace cylon{
namespace test{
int TestIndexBuildOperation(std::string &input_file_path, cylon::IndexingType indexing_type) {
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input1, output;
  std::shared_ptr<cylon::BaseArrowIndex> index;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << input_file_path << std::endl;
  status = cylon::FromCSV(ctx, input_file_path, input1, read_options);

  const int index_column = 0;

  status = cylon::IndexUtil::BuildArrowIndex(indexing_type, input1, index_column, index);

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

static int set_data_for_arrow_indexing_test(std::string &input_file_path,
									  cylon::IndexingType indexing_type,
									  std::string &output_file_path,
									  std::shared_ptr<cylon::Table> &input,
									  std::shared_ptr<cylon::Table> &expected_output,
									  std::shared_ptr<cylon::BaseArrowIndex> &index,
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

  status = cylon::IndexUtil::BuildArrowIndex(indexing_type, input, index_column, index);

  if (!status.is_ok()) {
	return 1;
  }

  bool valid_index = false;
  if (indexing_type == cylon::IndexingType::Range) {
	valid_index = input->Rows() == index->GetSize();
  } else {
	valid_index = input->Rows() == index->GetIndexArray()->length();
  }

  if (!valid_index) {
	return 1;
  };

  status = input->SetArrowIndex(index, drop_index);

  if (!status.is_ok()) {
	return 1;
  }

  return 0;
}


int TestIndexLocOperation1(std::string &input_file_path,
                           cylon::IndexingType indexing_type,
                           std::string &output_file_path) {
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;
  std::shared_ptr<cylon::Table> input, expected_output, result;
  std::shared_ptr<BaseArrowIndex> index;
  int res = set_data_for_arrow_indexing_test(input_file_path, indexing_type, output_file_path, input, expected_output, index, 1);

  if(res != 0) {
    return res;
  }

  //long start_index = 0;
  //long end_index = 5;
  int column = 0;

  auto start_index = arrow::MakeScalar<int64_t>(0);
  auto end_index = arrow::MakeScalar<int64_t>(5);

  std::shared_ptr<cylon::ArrowBaseIndexer> indexer = std::make_shared<cylon::ArrowLocIndexer>(indexing_type);

  status = indexer->loc(start_index, end_index, column, input, result);

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
                           cylon::IndexingType indexing_type,
                           std::string &output_file_path) {
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, expected_output, result;
  std::shared_ptr<cylon::BaseArrowIndex> index;
  int res = set_data_for_arrow_indexing_test(input_file_path, indexing_type, output_file_path, input, expected_output, index, 2);

  if(res != 0) {
    return res;
  }

//  long start_index = 0;
//  long end_index = 5;
  auto start_index = arrow::MakeScalar<int64_t>(0);
  auto end_index = arrow::MakeScalar<int64_t>(5);
  int start_column_index = 0;
  int end_column_index = 1;

  std::shared_ptr<cylon::ArrowBaseIndexer> indexer = std::make_shared<cylon::ArrowLocIndexer>(indexing_type);

  status = indexer->loc(start_index, end_index, start_column_index, end_column_index, input, result);

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
                           cylon::IndexingType indexing_type,
                           std::string &output_file_path) {
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, expected_output, result;
  std::shared_ptr<cylon::BaseArrowIndex> index;
  int res = set_data_for_arrow_indexing_test(input_file_path, indexing_type, output_file_path, input, expected_output, index, 3);

  if(res != 0) {
    return res;
  }

//  long start_index = 0;
//  long end_index = 5;
  auto start_index = arrow::MakeScalar<int64_t>(0);
  auto end_index = arrow::MakeScalar<int64_t>(5);
  std::vector<int> cols = {0,2};

  std::shared_ptr<cylon::ArrowBaseIndexer> indexer = std::make_shared<cylon::ArrowLocIndexer>(indexing_type);

  status = indexer->loc(start_index, end_index, cols, input, result);

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
                           cylon::IndexingType indexing_type,
                           std::string &output_file_path) {
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, expected_output, result;
  std::shared_ptr<cylon::BaseArrowIndex> index;
  int res = set_data_for_arrow_indexing_test(input_file_path, indexing_type, output_file_path, input, expected_output, index, 4);

  if(res != 0) {
    return res;
  }

  //long start_index = 10;
  auto pool = cylon::ToArrowPool(ctx);
  arrow::Int64Builder builder(pool);
  std::shared_ptr<arrow::Array> start_index;
  std::vector<int64_t> search_index_values = {10};
  builder.AppendValues(search_index_values);
  builder.Finish(&start_index);

  int column = 1;

  std::shared_ptr<cylon::ArrowBaseIndexer> indexer = std::make_shared<cylon::ArrowLocIndexer>(indexing_type);

  status = indexer->loc(start_index, column, input, result);

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
                           cylon::IndexingType indexing_type,
                           std::string &output_file_path) {
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, expected_output, result;
  std::shared_ptr<cylon::BaseArrowIndex> index;
  int res = set_data_for_arrow_indexing_test(input_file_path, indexing_type, output_file_path, input, expected_output, index, 5);

  if(res != 0) {
    return res;
  }

//  typedef std::vector<void *> vector_void_star;
//
//  vector_void_star output_items;
//
//  std::vector<long> start_indices = {4, 10};
  int start_column = 1;
  int end_column = 2;

//  for (size_t tx = 0; tx < start_indices.size(); tx++) {
//    long *val = new long(start_indices.at(tx));
//    output_items.push_back(static_cast<void *>(val));
//  }

  auto pool = cylon::ToArrowPool(ctx);
  arrow::Int64Builder builder(pool);
  std::shared_ptr<arrow::Array> start_index;
  std::vector<int64_t> search_index_values = {4, 10};
  builder.AppendValues(search_index_values);
  builder.Finish(&start_index);

  std::shared_ptr<cylon::ArrowBaseIndexer> indexer = std::make_shared<cylon::ArrowLocIndexer>(indexing_type);

  status = indexer->loc(start_index, start_column, end_column, input, result);

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
                           cylon::IndexingType indexing_type,
                           std::string &output_file_path) {
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, expected_output, result;
  std::shared_ptr<cylon::BaseArrowIndex> index;
  int res = set_data_for_arrow_indexing_test(input_file_path, indexing_type, output_file_path, input, expected_output, index, 6);

  if(res != 0) {
    return res;
  }

//  typedef std::vector<void *> vector_void_star;
//
//  vector_void_star output_items;
//
//  std::vector<long> start_indices = {4, 10};
  std::vector<int> columns = {0, 2};
  auto pool = cylon::ToArrowPool(ctx);
  arrow::Int64Builder builder(pool);
  std::shared_ptr<arrow::Array> start_index;
  std::vector<int64_t> search_index_values = {4, 10};
  builder.AppendValues(search_index_values);
  builder.Finish(&start_index);

//  for (size_t tx = 0; tx < start_indices.size(); tx++) {
//    long *val = new long(start_indices.at(tx));
//    output_items.push_back(static_cast<void *>(val));
//  }

  std::shared_ptr<cylon::ArrowBaseIndexer> indexer = std::make_shared<cylon::ArrowLocIndexer>(indexing_type);

  status = indexer->loc(start_index, columns, input, result);

  if (!status.is_ok()) {
    return -1;
  }

  if (test::Verify(ctx, result, expected_output)) {
    LOG(ERROR) << "Loc 6 failed!";
    return 1;
  }

  return 0;
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

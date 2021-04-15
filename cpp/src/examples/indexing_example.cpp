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

#include <net/mpi/mpi_communicator.hpp>
#include <ctx/cylon_context.hpp>
#include <table.hpp>
#include <ctx/arrow_memory_pool_utils.hpp>
#include <map>

#include "indexing/index_utils.hpp"
#include "indexing/indexer.hpp"

int arrow_take_test();

int indexing_simple_example();

void separator(std::string &statement);

int test_multi_map();

int build_int_index_from_values(std::shared_ptr<cylon::CylonContext> &ctx);

int build_str_index_from_values(std::shared_ptr<cylon::CylonContext> &ctx);

int build_array_from_vector(std::shared_ptr<cylon::CylonContext> &ctx);

int indexing_benchmark();

int test_hash_indexing();

int test_linear_indexing();

int test_iloc_operations();

int test_reset_index();

int test_loc_operations(cylon::IndexingType schema);

int test_str_loc_operations(cylon::IndexingType schema);

int print_arrow_array(std::shared_ptr<arrow::Array> &arr);

int print_index_output(std::shared_ptr<cylon::Table> &output, std::string &message);

int arrow_indexer_test_1();

int arrow_indexer_test_2();

int arrow_indexer_test_3();

int arrow_indexer_test_4();

int arrow_indexer_test_5();

int arrow_indexer_test_6();

int arrow_indexer_str_test_1();

int arrow_indexer_str_test_2();

int arrow_indexer_str_test_3();

int arrow_indexer_str_test_4();

int arrow_indexer_str_test_5();

int arrow_indexer_str_test_6();

int arrow_iloc_indexer_test_1();

int arrow_iloc_indexer_test_2();

int arrow_iloc_indexer_test_3();

int arrow_iloc_indexer_test_4();

int arrow_iloc_indexer_test_5();

int arrow_iloc_indexer_test_6();

int arrow_range_indexer_test();

int arrow_filter_example();

int create_int64_arrow_array(arrow::Int64Builder &builder,
							 int64_t capacity,
							 int64_t offset,
							 std::shared_ptr<arrow::Int64Array> &out_array);


//template<typename Base, typename T>
//inline bool instanceof(const T*);

/**
 * This example reads two csv files and does a union on them.
 * $ ./unique_example data.csv
 *
 * data.csv
 *  a,b,c,d
 *  1,2,3,2
    7,8,9,3
    10,11,12,4
    15,20,21,5
    10,11,24,6
    27,23,24,7
    1,2,13,8
    4,5,21,9
    39,23,24,10
    10,11,13,11
    123,11,12,12
    25,13,12,13
    30,21,22,14
    35,1,2,15
 */


int main(int argc, char *argv[]) {

  arrow_indexer_test_1();
  arrow_indexer_test_2();
  arrow_indexer_test_3();
  arrow_indexer_test_4();
  arrow_indexer_test_5();
  arrow_indexer_test_6();

  arrow_indexer_str_test_1();
  arrow_indexer_str_test_2();
  arrow_indexer_str_test_3();
  arrow_indexer_str_test_4();
  arrow_indexer_str_test_5();
  arrow_indexer_str_test_6();

  arrow_iloc_indexer_test_1();
  arrow_iloc_indexer_test_2();
  arrow_iloc_indexer_test_3();
  arrow_iloc_indexer_test_4();
  arrow_iloc_indexer_test_5();
  arrow_iloc_indexer_test_6();

  arrow_range_indexer_test();
  arrow_filter_example();

}

int arrow_take_test(std::shared_ptr<cylon::CylonContext> &ctx, std::shared_ptr<cylon::Table> &input1) {

  std::cout << "Arrow Take Test" << std::endl;

  std::shared_ptr<arrow::Array> out_idx;
  arrow::Status arrow_status;
  auto pool = cylon::ToArrowPool(ctx);
  arrow::compute::ExecContext fn_ctx(pool);
  arrow::Int64Builder idx_builder(pool);
  const arrow::Datum input_table(input1->get_table());

  idx_builder.AppendValues({0, 1, 3});
  arrow_status = idx_builder.Finish(&out_idx);

  const arrow::Datum filter_indices(out_idx);

  arrow::Result<arrow::Datum>
	  result = arrow::compute::Take(input_table, filter_indices, arrow::compute::TakeOptions::Defaults(), &fn_ctx);

  std::shared_ptr<arrow::Table> filter_table;
  std::shared_ptr<cylon::Table> ftb;
  filter_table = result.ValueOrDie().table();
  if (result.status() != arrow::Status::OK()) {
	std::cout << "Error occured in LocationByValue" << std::endl;
  } else {
	std::cout << "LocationByValue Succeeded" << std::endl;
  }
  cylon::Table::FromArrowTable(ctx, filter_table, ftb);

  ftb->Print();

  return 0;
}

void separator(std::string &statement) {
  std::cout << "===============================================" << std::endl;
  std::cout << statement << std::endl;
  std::cout << "===============================================" << std::endl;
}

int build_int_index_from_values(std::shared_ptr<cylon::CylonContext> &ctx) {
  arrow::Status arrow_status;

  std::shared_ptr<cylon::BaseArrowIndex> custom_index;
  std::shared_ptr<arrow::Array> index_values;
  std::vector<int32_t> ix_vals = {1, 2, 3, 4, 11};
  auto pool = cylon::ToArrowPool(ctx);
  arrow::Int32Builder int_32_builder(pool);

  int_32_builder.AppendValues(ix_vals);
  arrow_status = int_32_builder.Finish(&index_values);

  if (!arrow_status.ok()) {
	return -1;
  }

  cylon::IndexUtil::BuildArrowHashIndexFromArray(index_values, pool, custom_index);

  std::shared_ptr<arrow::Array> arr = custom_index->GetIndexAsArray();
  std::shared_ptr<arrow::Int32Scalar> int_32_scalar;

  for (int64_t xi = 0; xi < arr->length(); xi++) {
	auto result = arr->GetScalar(xi);
	if (!result.ok()) {
	  return -1;
	}
	auto scalar = result.ValueOrDie();
	int_32_scalar = std::static_pointer_cast<arrow::Int32Scalar>(scalar);
	std::cout << " " << int_32_scalar->value;
  }
  std::cout << std::endl;
  return 0;
}

int build_str_index_from_values(std::shared_ptr<cylon::CylonContext> &ctx) {
  arrow::Status arrow_status;

  std::shared_ptr<cylon::BaseArrowIndex> custom_index;
  std::shared_ptr<arrow::Array> index_values;
  std::vector<std::string> ix_vals = {"xp", "xq", "xr", "xs", "xt"};
  auto pool = cylon::ToArrowPool(ctx);
  arrow::StringBuilder string_builder(pool);

  arrow_status = string_builder.AppendValues(ix_vals);

  if (!arrow_status.ok()) {
	return -1;
  }

  arrow_status = string_builder.Finish(&index_values);

  if (!arrow_status.ok()) {
	return -1;
  }

  cylon::IndexUtil::BuildArrowHashIndexFromArray(index_values, pool, custom_index);

  std::shared_ptr<arrow::Array> arr = custom_index->GetIndexAsArray();
  std::shared_ptr<arrow::StringScalar> string_scalar;
  std::cout << "Str Array HashIndex length : " << arr->length() << std::endl;
  for (int64_t xi = 0; xi < arr->length(); xi++) {
	auto result = arr->GetScalar(xi);
	if (!result.ok()) {
	  return -1;
	}
	auto scalar = result.ValueOrDie();
	string_scalar = std::static_pointer_cast<arrow::StringScalar>(scalar);
	std::cout << " " << string_scalar->value->ToString();
  }
  std::cout << std::endl;
  return 0;
}

//int build_array_from_vector(std::shared_ptr<cylon::CylonContext> &ctx) {
//
//  LOG(INFO) << "BuildHashIndex index from vector";
//  std::vector<int> index_vector0{1, 2, 3, 4, 5};
//  std::vector<int16_t> index_vector{1, 2, 3, 4, 5};
//  std::vector<int32_t> index_vector1{1, 2, 3, 4, 5};
//  std::vector<int64_t> index_vector2{1, 2, 3, 4, 5};
//  std::vector<double_t> index_vector3{1, 2, 3, 4, 5};
//  std::vector<float_t> index_vector4{1, 2, 3, 4, 5};
//
//  std::shared_ptr<cylon::BaseArrowIndex> index;
//  auto pool = cylon::ToArrowPool(ctx);
//  cylon::IndexUtil::BuildArrowHashIndexFromVector(index_vector3, pool, index);
//
//  auto arr = index->GetIndexAsArray();
//  LOG(INFO) << "Array length : " << arr->length();
//
//  std::shared_ptr<arrow::DoubleScalar> int_32_scalar;
//
//  for (int64_t xi = 0; xi < arr->length(); xi++) {
//	auto result = arr->GetScalar(xi);
//	if (!result.ok()) {
//	  return -1;
//	}
//	auto scalar = result.ValueOrDie();
//	int_32_scalar = std::static_pointer_cast<arrow::DoubleScalar>(scalar);
//	std::cout << " " << int_32_scalar->value;
//  }
//  std::cout << std::endl;
//
//  return 0;
//}

//int test_loc_operations(cylon::IndexingSchema schema) {
//  std::string func_title = "[Numeric] Testing Indexing Schema " + std::to_string(schema);
//  separator(func_title);
//  LOG(INFO) << "Testing Indexing Schema " << schema;
//  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
//  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
//
//  cylon::Status status;
//
//  std::shared_ptr<cylon::Table> input, output, output1;
//  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);
//
//  // read first table
//  std::string test_file = "/tmp/indexing_data.csv";
//  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
//  status = cylon::FromCSV(ctx, test_file, input, read_options);
//
//  if (!status.is_ok()) {
//	LOG(ERROR) << "Error occurred in creating table";
//	return -1;
//  }
//
//  LOG(INFO) << "Original Data";
//
//  input->Print();
//
//  long start_index = 0;
//  long end_index = 5;
//  int64_t column = 0;
//  int start_column = 0;
//  int end_column = 1;
//  std::vector<int> columns = {0, 1};
//  typedef std::vector<void *> vector_void_star;
//
//  vector_void_star output_items;
//  long a = 4;
//  long b = 5;
//  std::vector<long> start_indices = {4, 1};
//  std::shared_ptr<arrow::Int64Array> arr;
//  arrow::Int64Builder builder;
//  builder.AppendValues(start_indices);
//  builder.Finish(&arr);
//
//  for (size_t tx = 0; tx < start_indices.size(); tx++) {
//	output_items.push_back(&start_indices.at(tx));
//  }
//
//  bool drop_index = true;
//  std::shared_ptr<arrow::Array> index_arr;
//
//  std::shared_ptr<cylon::BaseIndex> index;
//
//  status = cylon::IndexUtil::BuildIndex(schema, input, 0, index);
//
//  if (!status.is_ok()) {
//	return -1;
//  }
//
//  status = input->Set_Index(index, drop_index);
//
//  if (!status.is_ok()) {
//	return -1;
//  }
//
//  std::string loc_output_msg;
//
//  LOG(INFO) << "[RangeIndex] Records in Table Rows: " << input->Rows() << ", Columns: " << input->Columns();
//
//  std::shared_ptr<cylon::BaseIndexer> loc_indexer = std::make_shared<cylon::LocIndexer>(schema);
//
////  LOG(INFO) << "LOC Mode 1 Example";
////
////  loc_indexer->loc(&start_index, &end_index, column, input, output);
////
////  loc_output_msg = " LOC 1 ";
////  print_index_output(output, loc_output_msg);
////
////  LOG(INFO) << "LOC Mode 2 Example";
////
////  loc_indexer->loc(&start_index, &end_index, start_column, end_column, input, output);
////
////  loc_output_msg = " LOC 2 ";
////  print_index_output(output, loc_output_msg);
////
////  LOG(INFO) << "LOC Mode 3 Example";
////
////  loc_indexer->loc(&start_index, &end_index, columns, input, output);
////
////  loc_output_msg = " LOC 3 ";
////  print_index_output(output, loc_output_msg);
////
////  LOG(INFO) << "LOC Mode 4 Example";
////
////  loc_indexer->loc(&start_index, column, input, output);
////
////  loc_output_msg = " LOC 4 ";
////  print_index_output(output, loc_output_msg);
////
////  LOG(INFO) << "LOC Mode 5 Example";
////
////  loc_indexer->loc(&start_index, start_column, end_column, input, output);
////
////  loc_output_msg = " LOC 5 ";
////  print_index_output(output, loc_output_msg);
////
////  LOG(INFO) << "LOC Mode 6 Example";
////
////  loc_indexer->loc(&start_index, columns, input, output);
////
////  loc_output_msg = " LOC 6 ";
////  print_index_output(output, loc_output_msg);
//
//  LOG(INFO) << "LOC Mode 7 Example";
//  std::vector<void *> tvec{(void *)(arr->raw_values()), (void *)(arr->raw_values() + 1)};
//  loc_indexer->loc(tvec, column, input, output);
//
//  loc_output_msg = " LOC 7 ";
//  print_index_output(output, loc_output_msg);
//
////  LOG(INFO) << "LOC Mode 8 Example";
////
////  loc_indexer->loc(output_items, start_column, end_column, input, output);
////
////  loc_output_msg = " LOC 8 ";
////  print_index_output(output, loc_output_msg);
////
////  LOG(INFO) << "LOC Mode 9 Example";
////
////  loc_indexer->loc(output_items, columns, input, output);
////
////  loc_output_msg = " LOC 9 ";
////  print_index_output(output, loc_output_msg);
//
//  return 0;
//}

//int test_str_loc_operations(cylon::IndexingSchema schema) {
//  std::string func_title = "[String] Testing Indexing Schema " + std::to_string(schema);
//  separator(func_title);
//  LOG(INFO) << "Testing Indexing Schema " << schema;
//  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
//  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
//
//  cylon::Status status;
//
//  std::shared_ptr<cylon::Table> input, output, output1;
//  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);
//
//  // read first table
//  std::string test_file = "/home/vibhatha/build/cylon/data/input/indexing_str_data.csv";
//  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
//  status = cylon::FromCSV(ctx, test_file, input, read_options);
//
//  if (!status.is_ok()) {
//	LOG(ERROR) << "Error occurred in creating table";
//	return -1;
//  }
//
//  LOG(INFO) << "Original Data";
//
//  input->Print();
//
//  std::string start_index = "f";
//  std::string end_index = "m";
//  int64_t column = 0;
//  int start_column = 0;
//  int end_column = 1;
//  std::vector<int> columns = {0, 1};
//  typedef std::vector<void *> vector_void_star;
//
//  vector_void_star output_items;
//
//  std::vector<std::string> start_indices = {"e", "k"};
//  LOG(INFO) << "Filling";
//  for (size_t tx = 0; tx < start_indices.size(); tx++) {
//	std::string *sval = new std::string(start_indices.at(tx));
//	std::cout << *sval << ":" << sval << std::endl;
//	output_items.push_back(static_cast<void *>(&start_indices.at(tx)));
//  }
//  LOG(INFO) << "Loading";
//  for (size_t i = 0; i < output_items.size(); i++) {
//	void *pStr = output_items.at(i);
//	std::string &s = *(static_cast<std::string *>(pStr));
//	std::cout << s << ":" << &s << std::endl;
//  }
//  bool drop_index = true;
//
//  std::shared_ptr<cylon::BaseIndex> index;
//
//  status = cylon::IndexUtil::BuildIndex(schema, input, 0, index);
//
//  if (!status.is_ok()) {
//	return -1;
//  }
//
//  status = input->Set_Index(index, drop_index);
//
//  if (!status.is_ok()) {
//	return -1;
//  }
//
//  std::string loc_output_msg;
//
//  LOG(INFO) << "[RangeIndex] Records in Table Rows: " << input->Rows() << ", Columns: " << input->Columns();
//
//  std::shared_ptr<cylon::BaseIndexer> loc_indexer = std::make_shared<cylon::LocIndexer>(schema);
//
//  LOG(INFO) << "LOC Mode 1 Example";
//
//  loc_indexer->loc(&start_index, &end_index, column, input, output);
//
//  loc_output_msg = " LOC 1 ";
//  print_index_output(output, loc_output_msg);
//
//  LOG(INFO) << "LOC Mode 2 Example";
//
//  loc_indexer->loc(&start_index, &end_index, start_column, end_column, input, output);
//
//  loc_output_msg = " LOC 2 ";
//  print_index_output(output, loc_output_msg);
//
//  LOG(INFO) << "LOC Mode 3 Example";
//
//  loc_indexer->loc(&start_index, &end_index, columns, input, output);
//
//  loc_output_msg = " LOC 3 ";
//  print_index_output(output, loc_output_msg);
//
//  LOG(INFO) << "LOC Mode 4 Example";
//
//  loc_indexer->loc(&start_index, column, input, output);
//
//  loc_output_msg = " LOC 4 ";
//  print_index_output(output, loc_output_msg);
//
//  LOG(INFO) << "LOC Mode 5 Example";
//
//  loc_indexer->loc(&start_index, start_column, end_column, input, output);
//
//  loc_output_msg = " LOC 5 ";
//  print_index_output(output, loc_output_msg);
//
//  LOG(INFO) << "LOC Mode 6 Example";
//
//  loc_indexer->loc(&start_index, columns, input, output);
//
//  loc_output_msg = " LOC 6 ";
//  print_index_output(output, loc_output_msg);
//
//  LOG(INFO) << "LOC Mode 7 Example";
//
//  status = loc_indexer->loc(output_items, column, input, output);
//
//  if (!status.is_ok()) {
//	LOG(ERROR) << "Error occurred in operation LOC Mode 7";
//  }
//
//  loc_output_msg = " LOC 7 ";
//  print_index_output(output, loc_output_msg);
//
//  LOG(INFO) << "LOC Mode 8 Example";
//
//  status = loc_indexer->loc(output_items, start_column, end_column, input, output);
//
//  if (!status.is_ok()) {
//	LOG(ERROR) << "Error occurred in operation LOC Mode 8";
//  }
//
//  loc_output_msg = " LOC 8 ";
//  print_index_output(output, loc_output_msg);
//
//  LOG(INFO) << "LOC Mode 9 Example";
//
//  status = loc_indexer->loc(output_items, columns, input, output);
//
//  if (!status.is_ok()) {
//	LOG(ERROR) << "Error occurred in operation LOC Mode 9";
//  }
//
//  loc_output_msg = " LOC 9 ";
//  print_index_output(output, loc_output_msg);
//
//  return 0;
//}

//int test_iloc_operations() {
//  std::string func_title = "Testing ILoc ";
//  separator(func_title);
//  LOG(INFO) << func_title;
//  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
//  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
//
//  cylon::Status status;
//
//  std::shared_ptr<cylon::Table> input, output, output1;
//  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);
//
//  // read first table
//  std::string test_file = "/tmp/indexing_data.csv";
//  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
//  status = cylon::FromCSV(ctx, test_file, input, read_options);
//
//  if (!status.is_ok()) {
//	LOG(ERROR) << "Error occurred in creating table";
//	return -1;
//  }
//
//  LOG(INFO) << "Original Data";
//
//  input->Print();
//
//  long start_index = 0;
//  long end_index = 5;
//  int64_t column = 0;
//  int start_column = 0;
//  int end_column = 1;
//  std::vector<int> columns = {0, 1};
//  typedef std::vector<void *> vector_void_star;
//
//  vector_void_star output_items;
//
//  std::vector<long> start_indices = {4, 1};
//
//  for (size_t tx = 0; tx < start_indices.size(); tx++) {
//	long *val = new long(start_indices.at(tx));
//	output_items.push_back(static_cast<void *>(val));
//  }
//  bool drop_index = true;
//  std::string loc_output_msg;
//
//  std::shared_ptr<cylon::BaseIndex> index, loc_index, range_index;
//
//  status = cylon::IndexUtil::BuildIndex(cylon::IndexingSchema::Range, input, 0, range_index);
//
//  if (!status.is_ok()) {
//	return -1;
//  }
//
//  status = input->Set_Index(range_index, drop_index);
//
//  if (!status.is_ok()) {
//	return -1;
//  }
//
//  LOG(INFO) << "[RangeIndex] Records in Table Rows: " << input->Rows() << ", Columns: " << input->Columns();
//
//  std::shared_ptr<cylon::BaseIndexer> loc_indexer = std::make_shared<cylon::ILocIndexer>(cylon::IndexingSchema::Range);
//
//  LOG(INFO) << "iLOC Mode 1 Example";
//
//  loc_indexer->loc(&start_index, &end_index, column, input, output);
//
//  loc_output_msg = " iLOC 1 ";
//  print_index_output(output, loc_output_msg);
//
//  LOG(INFO) << "iLOC Mode 2 Example";
//
//  loc_indexer->loc(&start_index, &end_index, start_column, end_column, input, output);
//
//  loc_output_msg = " iLOC 1 ";
//  print_index_output(output, loc_output_msg);
//
//  LOG(INFO) << "iLOC Mode 3 Example";
//
//  loc_indexer->loc(&start_index, &end_index, columns, input, output);
//
//  loc_output_msg = " iLOC 1 ";
//  print_index_output(output, loc_output_msg);
//
//  LOG(INFO) << "iLOC Mode 4 Example";
//
//  loc_indexer->loc(&start_index, column, input, output);
//
//  loc_output_msg = " iLOC 1 ";
//  print_index_output(output, loc_output_msg);
//
//  LOG(INFO) << "iLOC Mode 5 Example";
//
//  loc_indexer->loc(&start_index, start_column, end_column, input, output);
//
//  loc_output_msg = " iLOC 1 ";
//  print_index_output(output, loc_output_msg);
//
//  LOG(INFO) << "iLOC Mode 6 Example";
//
//  loc_indexer->loc(&start_index, columns, input, output);
//
//  loc_output_msg = " iLOC 1 ";
//  print_index_output(output, loc_output_msg);
//
//  LOG(INFO) << "iLOC Mode 7 Example";
//
//  loc_indexer->loc(output_items, column, input, output);
//
//  loc_output_msg = " iLOC 1 ";
//  print_index_output(output, loc_output_msg);
//
//  LOG(INFO) << "iLOC Mode 8 Example";
//
//  loc_indexer->loc(output_items, start_column, end_column, input, output);
//
//  loc_output_msg = " iLOC 1 ";
//  print_index_output(output, loc_output_msg);
//
//  LOG(INFO) << "iLOC Mode 9 Example";
//
//  loc_indexer->loc(output_items, columns, input, output);
//
//  loc_output_msg = " iLOC 1 ";
//  print_index_output(output, loc_output_msg);
//
//  return 0;
//}

int print_arrow_array(std::shared_ptr<arrow::Array> &arr) {
  for (int64_t xi = 0; xi < arr->length(); xi++) {
	auto result = arr->GetScalar(xi);
	std::cout << " " << result.ValueOrDie()->ToString();
  }
  std::cout << std::endl;
  return 0;
}

//int print_index_output(std::shared_ptr<cylon::Table> &output, std::string &message) {
//  LOG(INFO) << message;
//  LOG(INFO) << "Loc operation Table";
//  output->Print();
//  LOG(INFO) << "Resultant Index";
//  auto index_arr = output->GetIndex()->GetIndexArray();
//  print_arrow_array(index_arr);
//  return 0;
//}

int test_scalar_casting() {
  std::unordered_multimap<int64_t, int64_t> map;

  map.emplace(10, 0);
  map.emplace(20, 1);
  map.emplace(5, 2);
  map.emplace(10, 3);
  map.emplace(20, 4);

  auto search_value = arrow::MakeScalar<int64_t>(10);
  auto casted_a = reinterpret_cast<arrow::Int64Scalar *>(search_value.get());

  auto search_value_1 = arrow::MakeScalar<int64_t>(20);
  std::shared_ptr<arrow::Int64Scalar>
	  casted_a_1 = std::static_pointer_cast<arrow::TypeTraits<arrow::Int64Type>::ScalarType>(search_value_1);

  auto search_val_str = arrow::MakeScalar("100");
  std::shared_ptr<arrow::StringScalar> casted_a_s = std::static_pointer_cast<arrow::StringScalar>(search_val_str);

  //auto result = search_value->CastTo(std::shared_ptr<arrow::Int64Type>());

  int64_t find_val = casted_a_1->value;

  std::cout << "Search Value in C : " << casted_a->value << ", " << find_val << ", "
			<< casted_a_s->value->ToString() << std::endl;

}

/*
 * Arrow Index based Loc Implementation
 * **/

int arrow_indexer_test_1() {
  std::string func_title = "Arrow Loc 1";
  separator(func_title);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output, output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table";
	return -1;
  }

  std::shared_ptr<cylon::Table> output_tb;

  auto start_idx = arrow::MakeScalar<int64_t>(7);
  auto end_idx = arrow::MakeScalar<int64_t>(1);

  std::cout << "Main Start Index : " << start_idx->ToString() << ", " << end_idx->ToString() << std::endl;

  std::shared_ptr<cylon::BaseArrowIndex> index;
  cylon::IndexingType schema = cylon::IndexingType::Hash;

  status = cylon::IndexUtil::BuildArrowIndex(schema, input, 0, true, output1);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating the Arrow Index";
  } else {
	LOG(INFO) << "Index Built Successfully!";
  }

  std::cout << "Output Table Index Schema : " << output1->GetArrowIndex()->GetIndexingType() << std::endl;
  std::cout << "Output Table Index Size : " << output1->GetArrowIndex()->GetSize() << std::endl;
  std::cout << "Output Table Index Array Size : " << output1->GetArrowIndex()->GetIndexArray()->length() << std::endl;

  auto index_as_array = output1->GetArrowIndex()->GetIndexAsArray();
  std::cout << "Index As an Array " << std::endl;

  print_arrow_array(index_as_array);

  std::shared_ptr<cylon::ArrowBaseIndexer>
	  loc_indexer = std::make_shared<cylon::ArrowLocIndexer>(schema);
  std::cout << "Creating Arrow Loc Indexer object" << std::endl;
  loc_indexer->loc(start_idx, end_idx, 0, output1, output_tb);

  output_tb->Print();

  auto index_arr = output_tb->GetArrowIndex()->GetIndexArray();

  std::cout << "Elements in Output Index : " << index_arr->length() << "[" << output_tb->GetArrowIndex()->GetSize()
			<< "]" << std::endl;

  print_arrow_array(index_arr);

  return 0;
}

int arrow_indexer_str_test_1() {
  std::string func_title = "Arrow Loc 1 [str]";
  separator(func_title);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output, output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_str_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table";
	return -1;
  }

  std::shared_ptr<cylon::Table> output_tb;

  auto start_idx = arrow::MakeScalar("f");
  auto end_idx = arrow::MakeScalar("k");

  std::cout << "Main Start Index : " << start_idx->ToString() << ", " << end_idx->ToString() << std::endl;

  std::shared_ptr<cylon::BaseArrowIndex> index;
  cylon::IndexingType schema = cylon::IndexingType::Hash;

  status = cylon::IndexUtil::BuildArrowIndex(schema, input, 0, true, output1);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating the Arrow Index";
  } else {
	LOG(INFO) << "Index Built Successfully!";
  }

  std::cout << "Output Table Index Schema : " << output1->GetArrowIndex()->GetIndexingType() << std::endl;
  std::cout << "Output Table Index Size : " << output1->GetArrowIndex()->GetSize() << std::endl;

  std::shared_ptr<cylon::ArrowBaseIndexer>
	  loc_indexer = std::make_shared<cylon::ArrowLocIndexer>(schema);
  std::cout << "Creating Arrow Loc Indexer object" << std::endl;
  loc_indexer->loc(start_idx, end_idx, 0, output1, output_tb);

  output_tb->Print();

  auto index_arr = output_tb->GetArrowIndex()->GetIndexArray();

  std::cout << "Elements in Output Index : " << index_arr->length() << "[" << output_tb->GetArrowIndex()->GetSize()
			<< "]" << std::endl;

  print_arrow_array(index_arr);

  return 0;
}

int arrow_indexer_test_2() {
  std::string func_title = "Arrow Loc 2";
  separator(func_title);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output, output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table";
	return -1;
  }

  std::shared_ptr<cylon::Table> output_tb;

  auto start_idx = arrow::MakeScalar<int64_t>(7);
  auto end_idx = arrow::MakeScalar<int64_t>(1);

  std::cout << "Main Start Index : " << start_idx->ToString() << ", " << end_idx->ToString() << std::endl;

  std::shared_ptr<cylon::BaseArrowIndex> index;
  cylon::IndexingType schema = cylon::IndexingType::Linear;

  int start_column_idx = 0;
  int end_column_idx = 1;

  status = cylon::IndexUtil::BuildArrowIndex(schema, input, 0, true, output1);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating the Arrow Index";
  } else {
	LOG(INFO) << "Index Built Successfully!";
  }

  std::cout << "Output Table Index Schema : " << output1->GetArrowIndex()->GetIndexingType() << std::endl;
  std::cout << "Output Table Index Size : " << output1->GetArrowIndex()->GetSize() << std::endl;

  std::shared_ptr<cylon::ArrowBaseIndexer>
	  loc_indexer = std::make_shared<cylon::ArrowLocIndexer>(schema);
  std::cout << "Creating Arrow Loc Indexer object" << std::endl;
  loc_indexer->loc(start_idx, end_idx, start_column_idx, end_column_idx, output1, output_tb);

  output_tb->Print();

  auto index_arr = output_tb->GetArrowIndex()->GetIndexArray();

  std::cout << "Elements in Output Index : " << index_arr->length() << "[" << output_tb->GetArrowIndex()->GetSize()
			<< "]" << std::endl;

  print_arrow_array(index_arr);

  return 0;
}

int arrow_indexer_str_test_2() {
  std::string func_title = "Arrow Loc 2 [str]";
  separator(func_title);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output, output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_str_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table";
	return -1;
  }

  std::shared_ptr<cylon::Table> output_tb;

  auto start_idx = arrow::MakeScalar("f");
  auto end_idx = arrow::MakeScalar("k");

  std::cout << "Main Start Index : " << start_idx->ToString() << ", " << end_idx->ToString() << std::endl;

  std::shared_ptr<cylon::BaseArrowIndex> index;
  cylon::IndexingType schema = cylon::IndexingType::Linear;

  int start_column_idx = 0;
  int end_column_idx = 1;

  status = cylon::IndexUtil::BuildArrowIndex(schema, input, 0, true, output1);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating the Arrow Index";
  } else {
	LOG(INFO) << "Index Built Successfully!";
  }

  std::cout << "Output Table Index Schema : " << output1->GetArrowIndex()->GetIndexingType() << std::endl;
  std::cout << "Output Table Index Size : " << output1->GetArrowIndex()->GetSize() << std::endl;

  std::shared_ptr<cylon::ArrowBaseIndexer>
	  loc_indexer = std::make_shared<cylon::ArrowLocIndexer>(schema);
  std::cout << "Creating Arrow Loc Indexer object" << std::endl;
  loc_indexer->loc(start_idx, end_idx, start_column_idx, end_column_idx, output1, output_tb);

  output_tb->Print();

  auto index_arr = output_tb->GetArrowIndex()->GetIndexArray();

  std::cout << "Elements in Output Index : " << index_arr->length() << "[" << output_tb->GetArrowIndex()->GetSize()
			<< "]" << std::endl;

  print_arrow_array(index_arr);

  return 0;
}

int arrow_indexer_test_3() {
  std::string func_title = "Arrow Loc 3";
  separator(func_title);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output, output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table";
	return -1;
  }

  std::shared_ptr<cylon::Table> output_tb;

  auto start_idx = arrow::MakeScalar<int64_t>(7);
  auto end_idx = arrow::MakeScalar<int64_t>(1);

  std::cout << "Main Start Index : " << start_idx->ToString() << ", " << end_idx->ToString() << std::endl;

  std::shared_ptr<cylon::BaseArrowIndex> index;
  cylon::IndexingType schema = cylon::IndexingType::Linear;

  std::vector<int> columns = {0, 1};

  status = cylon::IndexUtil::BuildArrowIndex(schema, input, 0, true, output1);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating the Arrow Index";
  } else {
	LOG(INFO) << "Index Built Successfully!";
  }

  std::cout << "Output Table Index Schema : " << output1->GetArrowIndex()->GetIndexingType() << std::endl;
  std::cout << "Output Table Index Size : " << output1->GetArrowIndex()->GetSize() << std::endl;

  std::shared_ptr<cylon::ArrowBaseIndexer>
	  loc_indexer = std::make_shared<cylon::ArrowLocIndexer>(schema);
  std::cout << "Creating Arrow Loc Indexer object" << std::endl;
  loc_indexer->loc(start_idx, end_idx, columns, output1, output_tb);

  output_tb->Print();

  auto index_arr = output_tb->GetArrowIndex()->GetIndexArray();

  std::cout << "Elements in Output Index : " << index_arr->length() << "[" << output_tb->GetArrowIndex()->GetSize()
			<< "]" << std::endl;

  print_arrow_array(index_arr);

  return 0;
}

int arrow_indexer_str_test_3() {
  std::string func_title = "Arrow Loc 3 [str]";
  separator(func_title);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output, output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_str_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table";
	return -1;
  }

  std::shared_ptr<cylon::Table> output_tb;

  auto start_idx = arrow::MakeScalar("f");
  auto end_idx = arrow::MakeScalar("k");

  std::cout << "Main Start Index : " << start_idx->ToString() << ", " << end_idx->ToString() << std::endl;

  std::shared_ptr<cylon::BaseArrowIndex> index;
  cylon::IndexingType schema = cylon::IndexingType::Linear;

  std::vector<int> columns = {0, 1};

  status = cylon::IndexUtil::BuildArrowIndex(schema, input, 0, true, output1);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating the Arrow Index";
  } else {
	LOG(INFO) << "Index Built Successfully!";
  }

  std::cout << "Output Table Index Schema : " << output1->GetArrowIndex()->GetIndexingType() << std::endl;
  std::cout << "Output Table Index Size : " << output1->GetArrowIndex()->GetSize() << std::endl;

  std::shared_ptr<cylon::ArrowBaseIndexer>
	  loc_indexer = std::make_shared<cylon::ArrowLocIndexer>(schema);
  std::cout << "Creating Arrow Loc Indexer object" << std::endl;
  loc_indexer->loc(start_idx, end_idx, columns, output1, output_tb);

  output_tb->Print();

  auto index_arr = output_tb->GetArrowIndex()->GetIndexArray();

  std::cout << "Elements in Output Index : " << index_arr->length() << "[" << output_tb->GetArrowIndex()->GetSize()
			<< "]" << std::endl;

  print_arrow_array(index_arr);

  return 0;
}

int arrow_indexer_test_4() {
  std::string func_title = "Arrow Loc 4";
  separator(func_title);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output, output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table";
	return -1;
  }

  std::shared_ptr<cylon::Table> output_tb;
  auto pool = cylon::ToArrowPool(ctx);
  arrow::Int64Builder builder(pool);
  std::shared_ptr<arrow::Array> search_index_array;
  std::vector<int64_t> search_index_values = {7, 10};
  builder.AppendValues(search_index_values);
  builder.Finish(&search_index_array);

  std::shared_ptr<cylon::BaseArrowIndex> index;
  cylon::IndexingType schema = cylon::IndexingType::Linear;

  status = cylon::IndexUtil::BuildArrowIndex(schema, input, 0, true, output1);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating the Arrow Index";
  } else {
	LOG(INFO) << "Index Built Successfully!";
  }

  std::cout << "Output Table Index Schema : " << output1->GetArrowIndex()->GetIndexingType() << std::endl;
  std::cout << "Output Table Index Size : " << output1->GetArrowIndex()->GetSize() << std::endl;

  std::shared_ptr<cylon::ArrowBaseIndexer>
	  loc_indexer = std::make_shared<cylon::ArrowLocIndexer>(schema);
  std::cout << "Creating Arrow Loc Indexer object" << std::endl;
  loc_indexer->loc(search_index_array, 0, output1, output_tb);

  output_tb->Print();

  auto index_arr = output_tb->GetArrowIndex()->GetIndexArray();

  std::cout << "Elements in Output Index : " << index_arr->length() << "[" << output_tb->GetArrowIndex()->GetSize()
			<< "]" << std::endl;

  print_arrow_array(index_arr);

  return 0;
}

int arrow_indexer_str_test_4() {
  std::string func_title = "Arrow Loc 4 [str]";
  separator(func_title);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output, output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_str_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table";
	return -1;
  }

  std::shared_ptr<cylon::Table> output_tb;
  auto pool = cylon::ToArrowPool(ctx);
  arrow::StringBuilder builder(pool);
  std::shared_ptr<arrow::Array> search_index_array;
  std::vector<std::string> search_index_values = {"f", "h"};
  builder.AppendValues(search_index_values);
  builder.Finish(&search_index_array);

  std::shared_ptr<cylon::BaseArrowIndex> index;
  cylon::IndexingType schema = cylon::IndexingType::Linear;

  status = cylon::IndexUtil::BuildArrowIndex(schema, input, 0, true, output1);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating the Arrow Index";
  } else {
	LOG(INFO) << "Index Built Successfully!";
  }

  std::cout << "Output Table Index Schema : " << output1->GetArrowIndex()->GetIndexingType() << std::endl;
  std::cout << "Output Table Index Size : " << output1->GetArrowIndex()->GetSize() << std::endl;

  std::shared_ptr<cylon::ArrowBaseIndexer>
	  loc_indexer = std::make_shared<cylon::ArrowLocIndexer>(schema);
  std::cout << "Creating Arrow Loc Indexer object" << std::endl;
  loc_indexer->loc(search_index_array, 0, output1, output_tb);

  output_tb->Print();

  auto index_arr = output_tb->GetArrowIndex()->GetIndexArray();

  std::cout << "Elements in Output Index : " << index_arr->length() << "[" << output_tb->GetArrowIndex()->GetSize()
			<< "]" << std::endl;

  print_arrow_array(index_arr);

  return 0;
}

int arrow_indexer_test_5() {
  std::string func_title = "Arrow Loc 5";
  separator(func_title);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output, output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table";
	return -1;
  }

  std::shared_ptr<cylon::Table> output_tb;
  auto pool = cylon::ToArrowPool(ctx);
  arrow::Int64Builder builder(pool);
  std::shared_ptr<arrow::Array> search_index_array;
  std::vector<int64_t> search_index_values = {7, 10};
  builder.AppendValues(search_index_values);
  builder.Finish(&search_index_array);
  int start_column_idx = 0;
  int end_column_idx = 1;

  std::shared_ptr<cylon::BaseArrowIndex> index;
  cylon::IndexingType schema = cylon::IndexingType::Linear;

  status = cylon::IndexUtil::BuildArrowIndex(schema, input, 0, true, output1);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating the Arrow Index";
  } else {
	LOG(INFO) << "Index Built Successfully!";
  }

  std::cout << "Output Table Index Schema : " << output1->GetArrowIndex()->GetIndexingType() << std::endl;
  std::cout << "Output Table Index Size : " << output1->GetArrowIndex()->GetSize() << std::endl;

  std::shared_ptr<cylon::ArrowBaseIndexer>
	  loc_indexer = std::make_shared<cylon::ArrowLocIndexer>(schema);
  std::cout << "Creating Arrow Loc Indexer object" << std::endl;
  loc_indexer->loc(search_index_array, start_column_idx, end_column_idx, output1, output_tb);

  output_tb->Print();

  auto index_arr = output_tb->GetArrowIndex()->GetIndexArray();

  std::cout << "Elements in Output Index : " << index_arr->length() << "[" << output_tb->GetArrowIndex()->GetSize()
			<< "]" << std::endl;

  print_arrow_array(index_arr);

  return 0;
}

int arrow_indexer_str_test_5() {
  std::string func_title = "Arrow Loc 5 [str]";
  separator(func_title);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output, output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_str_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table";
	return -1;
  }

  std::shared_ptr<cylon::Table> output_tb;
  auto pool = cylon::ToArrowPool(ctx);
  arrow::StringBuilder builder(pool);
  std::shared_ptr<arrow::Array> search_index_array;
  std::vector<std::string> search_index_values = {"f", "h"};
  builder.AppendValues(search_index_values);
  builder.Finish(&search_index_array);
  int start_column_idx = 0;
  int end_column_idx = 1;

  std::shared_ptr<cylon::BaseArrowIndex> index;
  cylon::IndexingType schema = cylon::IndexingType::Linear;

  status = cylon::IndexUtil::BuildArrowIndex(schema, input, 0, true, output1);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating the Arrow Index";
  } else {
	LOG(INFO) << "Index Built Successfully!";
  }

  std::cout << "Output Table Index Schema : " << output1->GetArrowIndex()->GetIndexingType() << std::endl;
  std::cout << "Output Table Index Size : " << output1->GetArrowIndex()->GetSize() << std::endl;

  std::shared_ptr<cylon::ArrowBaseIndexer>
	  loc_indexer = std::make_shared<cylon::ArrowLocIndexer>(schema);
  std::cout << "Creating Arrow Loc Indexer object" << std::endl;
  loc_indexer->loc(search_index_array, start_column_idx, end_column_idx, output1, output_tb);

  output_tb->Print();

  auto index_arr = output_tb->GetArrowIndex()->GetIndexArray();

  std::cout << "Elements in Output Index : " << index_arr->length() << "[" << output_tb->GetArrowIndex()->GetSize()
			<< "]" << std::endl;

  print_arrow_array(index_arr);

  return 0;
}

int arrow_indexer_test_6() {
  std::string func_title = "Arrow Loc 6";
  separator(func_title);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output, output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table";
	return -1;
  }

  std::shared_ptr<cylon::Table> output_tb;
  auto pool = cylon::ToArrowPool(ctx);
  arrow::Int64Builder builder(pool);
  std::shared_ptr<arrow::Array> search_index_array;
  std::vector<int64_t> search_index_values = {7, 10};
  builder.AppendValues(search_index_values);
  builder.Finish(&search_index_array);
  std::vector<int> columns = {0, 1};

  std::shared_ptr<cylon::BaseArrowIndex> index;
  cylon::IndexingType schema = cylon::IndexingType::Linear;

  status = cylon::IndexUtil::BuildArrowIndex(schema, input, 0, true, output1);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating the Arrow Index";
  } else {
	LOG(INFO) << "Index Built Successfully!";
  }

  std::cout << "Output Table Index Schema : " << output1->GetArrowIndex()->GetIndexingType() << std::endl;
  std::cout << "Output Table Index Size : " << output1->GetArrowIndex()->GetSize() << std::endl;

  std::shared_ptr<cylon::ArrowBaseIndexer>
	  loc_indexer = std::make_shared<cylon::ArrowLocIndexer>(schema);
  std::cout << "Creating Arrow Loc Indexer object" << std::endl;
  loc_indexer->loc(search_index_array, columns, output1, output_tb);

  output_tb->Print();

  auto index_arr = output_tb->GetArrowIndex()->GetIndexArray();

  std::cout << "Elements in Output Index : " << index_arr->length() << "[" << output_tb->GetArrowIndex()->GetSize()
			<< "]" << std::endl;

  print_arrow_array(index_arr);

  return 0;
}

int arrow_indexer_str_test_6() {
  std::string func_title = "Arrow Loc 6 [str]";
  separator(func_title);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output, output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_str_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table";
	return -1;
  }

  std::shared_ptr<cylon::Table> output_tb;
  auto pool = cylon::ToArrowPool(ctx);
  arrow::StringBuilder builder(pool);
  std::shared_ptr<arrow::Array> search_index_array;
  std::vector<std::string> search_index_values = {"f", "h"};
  builder.AppendValues(search_index_values);
  builder.Finish(&search_index_array);
  std::vector<int> columns = {0, 1};

  std::shared_ptr<cylon::BaseArrowIndex> index;
  cylon::IndexingType schema = cylon::IndexingType::Linear;

  status = cylon::IndexUtil::BuildArrowIndex(schema, input, 0, true, output1);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating the Arrow Index";
  } else {
	LOG(INFO) << "Index Built Successfully!";
  }

  std::cout << "Output Table Index Schema : " << output1->GetArrowIndex()->GetIndexingType() << std::endl;
  std::cout << "Output Table Index Size : " << output1->GetArrowIndex()->GetSize() << std::endl;

  std::shared_ptr<cylon::ArrowBaseIndexer>
	  loc_indexer = std::make_shared<cylon::ArrowLocIndexer>(schema);
  std::cout << "Creating Arrow Loc Indexer object" << std::endl;
  loc_indexer->loc(search_index_array, columns, output1, output_tb);

  output_tb->Print();

  auto index_arr = output_tb->GetArrowIndex()->GetIndexArray();

  std::cout << "Elements in Output Index : " << index_arr->length() << "[" << output_tb->GetArrowIndex()->GetSize()
			<< "]" << std::endl;

  print_arrow_array(index_arr);

  return 0;
}

/*
 * Arrow ILoc Operations
 * */


int arrow_iloc_indexer_test_1() {
  std::string func_title = "Arrow ILoc 1";
  separator(func_title);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output, output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table";
	return -1;
  }

  std::shared_ptr<cylon::Table> output_tb;

  auto start_idx = arrow::MakeScalar<int64_t>(0);
  auto end_idx = arrow::MakeScalar<int64_t>(5);

  std::cout << "Main Start Index : " << start_idx->ToString() << ", " << end_idx->ToString() << std::endl;

  std::shared_ptr<cylon::BaseArrowIndex> index;
  cylon::IndexingType schema = cylon::IndexingType::Range;

  status = cylon::IndexUtil::BuildArrowIndex(schema, input, 0, false, output1);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating the Arrow Index";
  } else {
	LOG(INFO) << "Index Built Successfully!";
  }

  std::cout << "Output Table Index Schema : " << output1->GetArrowIndex()->GetIndexingType() << std::endl;
  std::cout << "Output Table Index Size : " << output1->GetArrowIndex()->GetSize() << std::endl;

  std::shared_ptr<cylon::ArrowBaseIndexer>
	  loc_indexer = std::make_shared<cylon::ArrowILocIndexer>(schema);
  std::cout << "Creating Arrow Loc Indexer object" << std::endl;
  loc_indexer->loc(start_idx, end_idx, 0, output1, output_tb);

  output_tb->Print();

  auto index_arr = output_tb->GetArrowIndex()->GetIndexArray();

  std::cout << "Elements in Output Index : " << index_arr->length() << "[" << output_tb->GetArrowIndex()->GetSize()
			<< "]" << std::endl;

  print_arrow_array(index_arr);

  return 0;
}

int arrow_iloc_indexer_test_2() {
  std::string func_title = "Arrow ILoc 2";
  separator(func_title);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output, output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table";
	return -1;
  }

  std::shared_ptr<cylon::Table> output_tb;

  auto start_idx = arrow::MakeScalar<int64_t>(0);
  auto end_idx = arrow::MakeScalar<int64_t>(5);

  int start_column_idx = 0;
  int end_column_idx = 1;

  std::cout << "Main Start Index : " << start_idx->ToString() << ", " << end_idx->ToString() << std::endl;

  std::shared_ptr<cylon::BaseArrowIndex> index;
  cylon::IndexingType schema = cylon::IndexingType::Range;

  status = cylon::IndexUtil::BuildArrowIndex(schema, input, 0, false, output1);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating the Arrow Index";
  } else {
	LOG(INFO) << "Index Built Successfully!";
  }

  std::cout << "Output Table Index Schema : " << output1->GetArrowIndex()->GetIndexingType() << std::endl;
  std::cout << "Output Table Index Size : " << output1->GetArrowIndex()->GetSize() << std::endl;

  std::shared_ptr<cylon::ArrowBaseIndexer>
	  loc_indexer = std::make_shared<cylon::ArrowILocIndexer>(schema);
  std::cout << "Creating Arrow Loc Indexer object" << std::endl;
  loc_indexer->loc(start_idx, end_idx, start_column_idx, end_column_idx, output1, output_tb);

  output_tb->Print();

  auto index_arr = output_tb->GetArrowIndex()->GetIndexArray();

  std::cout << "Elements in Output Index : " << index_arr->length() << "[" << output_tb->GetArrowIndex()->GetSize()
			<< "]" << std::endl;

  print_arrow_array(index_arr);

  return 0;
}

int arrow_iloc_indexer_test_3() {
  std::string func_title = "Arrow ILoc 3";
  separator(func_title);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output, output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table";
	return -1;
  }

  std::shared_ptr<cylon::Table> output_tb;

  auto start_idx = arrow::MakeScalar<int64_t>(0);
  auto end_idx = arrow::MakeScalar<int64_t>(5);

  std::vector<int> columns = {0, 1};

  std::cout << "Main Start Index : " << start_idx->ToString() << ", " << end_idx->ToString() << std::endl;

  std::shared_ptr<cylon::BaseArrowIndex> index;
  cylon::IndexingType schema = cylon::IndexingType::Range;

  status = cylon::IndexUtil::BuildArrowIndex(schema, input, 0, false, output1);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating the Arrow Index";
  } else {
	LOG(INFO) << "Index Built Successfully!";
  }

  std::cout << "Output Table Index Schema : " << output1->GetArrowIndex()->GetIndexingType() << std::endl;
  std::cout << "Output Table Index Size : " << output1->GetArrowIndex()->GetSize() << std::endl;

  std::shared_ptr<cylon::ArrowBaseIndexer>
	  loc_indexer = std::make_shared<cylon::ArrowILocIndexer>(schema);
  std::cout << "Creating Arrow Loc Indexer object" << std::endl;
  loc_indexer->loc(start_idx, end_idx, columns, output1, output_tb);

  output_tb->Print();

  auto index_arr = output_tb->GetArrowIndex()->GetIndexArray();

  std::cout << "Elements in Output Index : " << index_arr->length() << "[" << output_tb->GetArrowIndex()->GetSize()
			<< "]" << std::endl;

  print_arrow_array(index_arr);

  return 0;
}

int arrow_iloc_indexer_test_4() {
  std::string func_title = "Arrow ILoc 4";
  separator(func_title);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output, output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table";
	return -1;
  }

  std::shared_ptr<cylon::Table> output_tb;
  auto pool = cylon::ToArrowPool(ctx);
  arrow::Int64Builder builder(pool);
  std::shared_ptr<arrow::Array> search_index_array;
  std::vector<int64_t> search_index_values = {0, 1, 2, 3, 4, 5};
  builder.AppendValues(search_index_values);
  builder.Finish(&search_index_array);

  std::shared_ptr<cylon::BaseArrowIndex> index;
  cylon::IndexingType schema = cylon::IndexingType::Range;

  status = cylon::IndexUtil::BuildArrowIndex(schema, input, 0, false, output1);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating the Arrow Index";
  } else {
	LOG(INFO) << "Index Built Successfully!";
  }

  std::cout << "Output Table Index Schema : " << output1->GetArrowIndex()->GetIndexingType() << std::endl;
  std::cout << "Output Table Index Size : " << output1->GetArrowIndex()->GetSize() << std::endl;

  std::shared_ptr<cylon::ArrowBaseIndexer>
	  loc_indexer = std::make_shared<cylon::ArrowILocIndexer>(schema);
  std::cout << "Creating Arrow Loc Indexer object" << std::endl;
  loc_indexer->loc(search_index_array, 0, output1, output_tb);

  output_tb->Print();

  auto index_arr = output_tb->GetArrowIndex()->GetIndexArray();

  std::cout << "Elements in Output Index : " << index_arr->length() << "[" << output_tb->GetArrowIndex()->GetSize()
			<< "]" << std::endl;

  print_arrow_array(index_arr);

  return 0;
}

int arrow_iloc_indexer_test_5() {
  std::string func_title = "Arrow ILoc 5";
  separator(func_title);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output, output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table";
	return -1;
  }

  std::shared_ptr<cylon::Table> output_tb;
  auto pool = cylon::ToArrowPool(ctx);
  arrow::Int64Builder builder(pool);
  std::shared_ptr<arrow::Array> search_index_array;
  std::vector<int64_t> search_index_values = {0, 1, 2, 3, 4, 5};
  builder.AppendValues(search_index_values);
  builder.Finish(&search_index_array);

  int start_column_idx = 0;
  int end_column_idx = 1;

  std::shared_ptr<cylon::BaseArrowIndex> index;
  cylon::IndexingType schema = cylon::IndexingType::Range;

  status = cylon::IndexUtil::BuildArrowIndex(schema, input, 0, false, output1);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating the Arrow Index";
  } else {
	LOG(INFO) << "Index Built Successfully!";
  }

  std::cout << "Output Table Index Schema : " << output1->GetArrowIndex()->GetIndexingType() << std::endl;
  std::cout << "Output Table Index Size : " << output1->GetArrowIndex()->GetSize() << std::endl;

  std::shared_ptr<cylon::ArrowBaseIndexer>
	  loc_indexer = std::make_shared<cylon::ArrowILocIndexer>(schema);
  std::cout << "Creating Arrow Loc Indexer object" << std::endl;
  loc_indexer->loc(search_index_array, start_column_idx, end_column_idx, output1, output_tb);

  output_tb->Print();

  auto index_arr = output_tb->GetArrowIndex()->GetIndexArray();

  std::cout << "Elements in Output Index : " << index_arr->length() << "[" << output_tb->GetArrowIndex()->GetSize()
			<< "]" << std::endl;

  print_arrow_array(index_arr);

  return 0;
}

int arrow_iloc_indexer_test_6() {
  std::string func_title = "Arrow ILoc 6";
  separator(func_title);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output, output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table";
	return -1;
  }

  std::shared_ptr<cylon::Table> output_tb;
  auto pool = cylon::ToArrowPool(ctx);
  arrow::Int64Builder builder(pool);
  std::shared_ptr<arrow::Array> search_index_array;
  std::vector<int64_t> search_index_values = {0, 1, 2, 3, 4, 5};
  builder.AppendValues(search_index_values);
  builder.Finish(&search_index_array);

  std::vector<int> columns = {0, 1};

  std::shared_ptr<cylon::BaseArrowIndex> index;
  cylon::IndexingType schema = cylon::IndexingType::Range;

  status = cylon::IndexUtil::BuildArrowIndex(schema, input, 0, false, output1);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating the Arrow Index";
  } else {
	LOG(INFO) << "Index Built Successfully!";
  }

  std::cout << "Output Table Index Schema : " << output1->GetArrowIndex()->GetIndexingType() << std::endl;
  std::cout << "Output Table Index Size : " << output1->GetArrowIndex()->GetSize() << std::endl;

  std::shared_ptr<cylon::ArrowBaseIndexer>
	  loc_indexer = std::make_shared<cylon::ArrowILocIndexer>(schema);
  std::cout << "Creating Arrow Loc Indexer object" << std::endl;
  loc_indexer->loc(search_index_array, columns, output1, output_tb);

  output_tb->Print();

  auto index_arr = output_tb->GetArrowIndex()->GetIndexArray();

  std::cout << "Elements in Output Index : " << index_arr->length() << "[" << output_tb->GetArrowIndex()->GetSize()
			<< "]" << std::endl;

  print_arrow_array(index_arr);

  return 0;
}

int create_int64_arrow_array(arrow::Int64Builder &builder,
							 int64_t capacity,
							 int64_t offset,
							 std::shared_ptr<arrow::Int64Array> &out_array) {

  builder.Reserve(capacity);
  for (int64_t ix = 0 + offset; ix < capacity + offset; ix++) {
	builder.Append(ix);
  }
  builder.Finish(&out_array);

  return 0;
}

int arrow_filter_example() {
  std::string func_title = "Arrow Filter Test";
  separator(func_title);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output;
  int64_t capacity = 10;
  int64_t search_capacity = 4;
  int64_t offset = 3;
  std::shared_ptr<cylon::Table> output_tb;
  std::shared_ptr<arrow::Int64Array> search_index_array, data_array, other_data;
  auto pool = cylon::ToArrowPool(ctx);
//  arrow::Int64Builder builder1(pool);
//  arrow::Int64Builder builder2(pool);
//  arrow::Int64Builder builder3(pool);

//  create_int64_arrow_array(builder1, capacity, 0, data_array);
//  create_int64_arrow_array(builder2, search_capacity, offset, search_index_array);
//  create_int64_arrow_array(builder3, capacity, 100, other_data);
//
//  LOG(INFO) << "Data Array";
//  auto casted_data_array = std::static_pointer_cast<arrow::Array>(data_array);
//  print_arrow_array(casted_data_array);
//  LOG(INFO) << "Search Array";
//  auto casted_search_array = std::static_pointer_cast<arrow::Array>(search_index_array);
//  print_arrow_array(casted_search_array);
//  LOG(INFO) << "Other Array";
//  auto casted_other_array = std::static_pointer_cast<arrow::Array>(other_data);
//  print_arrow_array(casted_other_array);

//  auto res_isin_filter = arrow::compute::IsIn(data_array, search_index_array);
//
//  if (res_isin_filter.ok()) {
//	std::cout << "Successfully Filtered!!!" << std::endl;
//  } else {
//	std::cout << "Failed Filtering... :/" << std::endl;
//  }
//
//  auto res_isin_filter_val = res_isin_filter.ValueOrDie();
//
//  if (res_isin_filter_val.is_array()) {
//	std::cout << "Filter response is an array" << std::endl;
//  } else {
//	std::cout << "Filter response is not an array" << std::endl;
//  }
//
//  std::shared_ptr<arrow::ArrayData>
//	  filtered_isin_array = std::static_pointer_cast<arrow::ArrayData>(res_isin_filter_val.array());
//
//  std::shared_ptr<arrow::ChunkedArray>
//	  cr_isin = std::make_shared<arrow::ChunkedArray>(arrow::MakeArray(filtered_isin_array));
//
//  std::shared_ptr<arrow::Array> arr_isin = cr_isin->chunk(0);
//
//  print_arrow_array(arr_isin);
//
//  auto filter_1 = arrow::compute::Filter(other_data, arr_isin).ValueOrDie();
//
//  std::shared_ptr<arrow::ArrayData> fil_res_array = std::static_pointer_cast<arrow::ArrayData>(filter_1.array());
//
//  std::shared_ptr<arrow::ChunkedArray> cr1 = std::make_shared<arrow::ChunkedArray>(arrow::MakeArray(fil_res_array));
//
//  std::shared_ptr<arrow::Array> arr1 = cr1->chunk(0);
//
//  print_arrow_array(arr1);

  ////////////////////////

  std::shared_ptr<cylon::Table> output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table";
	return -1;
  }

  arrow::Int64Builder builder(pool);
  std::vector<int64_t> search_index_values = {7, 10};
  builder.AppendValues(search_index_values);
  builder.Finish(&search_index_array);

  std::vector<int> columns = {0, 1};

  std::shared_ptr<cylon::BaseArrowIndex> index;
  cylon::IndexingType schema = cylon::IndexingType::Linear;

  status = cylon::IndexUtil::BuildArrowIndex(schema, input, 0, true, output1);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating the Arrow Index";
  } else {
	LOG(INFO) << "Index Built Successfully!";
  }

  std::cout << "Output Table Index Schema : " << output1->GetArrowIndex()->GetIndexingType() << std::endl;
  std::cout << "Output Table Index Size : " << output1->GetArrowIndex()->GetSize() << std::endl;

  auto index_array_ = output1->GetArrowIndex()->GetIndexArray();

  auto cast_search_param_result = arrow::compute::Cast(search_index_array, index_array_->type());

  std::shared_ptr<arrow::ArrayData> cast_search_param = cast_search_param_result.ValueOrDie().array();

  std::shared_ptr<arrow::ChunkedArray>
	  cast_search_param_chr = std::make_shared<arrow::ChunkedArray>(arrow::MakeArray(cast_search_param));

  auto search_param_array = cast_search_param_chr->chunk(0);

  auto res_isin_filter = arrow::compute::IsIn(index_array_, search_param_array);

  if (res_isin_filter.ok()) {
	std::cout << "Successfully Filtered!!!" << std::endl;
  } else {
	std::cout << "Failed Filtering... :/" << std::endl;
  }

  auto res_isin_filter_val = res_isin_filter.ValueOrDie();

  if (res_isin_filter_val.is_array()) {
	std::cout << "Filter response is an array" << std::endl;
  } else {
	std::cout << "Filter response is not an array" << std::endl;
  }

  std::shared_ptr<arrow::ArrayData>
	  filtered_isin_array = std::static_pointer_cast<arrow::ArrayData>(res_isin_filter_val.array());

  std::shared_ptr<arrow::ChunkedArray>
	  cr_isin = std::make_shared<arrow::ChunkedArray>(arrow::MakeArray(filtered_isin_array));

  std::shared_ptr<arrow::Array> arr_isin = cr_isin->chunk(0);

  print_arrow_array(arr_isin);

  std::shared_ptr<arrow::BooleanArray> arr_isin_bool_array = std::static_pointer_cast<arrow::BooleanArray>(arr_isin);

  arrow::Int64Builder filter_index_builder(pool);
  std::shared_ptr<arrow::Int64Array> filter_index_array;
  filter_index_builder.Reserve(arr_isin->length());
  auto bool_scalar_true = arrow::MakeScalar<bool>(true);

  std::shared_ptr<arrow::BooleanScalar>
	  boolean_scalar = std::static_pointer_cast<arrow::BooleanScalar>(bool_scalar_true);

  std::cout << "Make Bool Scalar : " << bool_scalar_true->ToString() << std::endl;

  for (int64_t ix = 0; ix < arr_isin_bool_array->length(); ix++) {

	auto val = arr_isin_bool_array->Value(ix);
	if (val) {
	  filter_index_builder.Append(ix);
	}
  }
  filter_index_builder.Finish(&filter_index_array);
  std::shared_ptr<arrow::Array> casted_index_array = std::static_pointer_cast<arrow::Array>(filter_index_array);
  print_arrow_array(casted_index_array);

  return 0;
}

int arrow_range_indexer_test() {
  std::string func_title = "Range Test case";
  separator(func_title);

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output, output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_data_test.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating table";
	return -1;
  }

  std::shared_ptr<cylon::Table> output_tb;
  auto pool = cylon::ToArrowPool(ctx);
  arrow::Int64Builder builder(pool);
  std::shared_ptr<arrow::Array> search_index_array;
  std::vector<int64_t> search_index_values = {10};
  builder.AppendValues(search_index_values);
  builder.Finish(&search_index_array);

  std::shared_ptr<cylon::BaseArrowIndex> index;
  cylon::IndexingType schema = cylon::IndexingType::Range;

  status = cylon::IndexUtil::BuildArrowIndex(schema, input, 0, true, output1);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in creating the Arrow Index";
  } else {
	LOG(INFO) << "Index Built Successfully!";
  }

  std::cout << "Output Table Index Schema : " << output1->GetArrowIndex()->GetIndexingType() << std::endl;
  std::cout << "Output Table Index Size : " << output1->GetArrowIndex()->GetSize() << std::endl;
  std::cout << "Output Table Array Size : " << output1->GetArrowIndex()->GetIndexArray()->length() << std::endl;

  std::shared_ptr<cylon::ArrowBaseIndexer>
	  loc_indexer = std::make_shared<cylon::ArrowLocIndexer>(schema);
  std::cout << "Creating Arrow Loc Indexer object" << std::endl;
  loc_indexer->loc(search_index_array, 1, output1, output_tb);
  std::cout << "Output ===>" << std::endl;
  output_tb->Print();

  auto index_arr = output_tb->GetArrowIndex()->GetIndexArray();

  std::cout << "Elements in Output Index : " << index_arr->length() << "[" << output_tb->GetArrowIndex()->GetSize()
			<< "]" << std::endl;

  print_arrow_array(index_arr);

  return 0;
}
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
#include <typeinfo>

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

int test_loc_operations(cylon::IndexingSchema schema);

int test_str_loc_operations(cylon::IndexingSchema schema);

int test_range_indexing();

int print_arrow_array(std::shared_ptr<arrow::Array> &arr);


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
  //indexing_simple_example();
  //test_hash_indexing();
  //test_linear_indexing();
  //indexing_benchmark();
  //test_range_indexing();
  std::vector<cylon::IndexingSchema> schemas{cylon::IndexingSchema::Range, cylon::IndexingSchema::Linear, cylon::Hash};
  for (auto schema : schemas) {
    test_loc_operations(schema);
  }

  for (size_t i = 1; i < schemas.size(); i++) {
    test_str_loc_operations(schemas.at(i));
  }

  test_iloc_operations();


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

int indexing_simple_example() {
  std::cout << "Dummy Test" << std::endl;
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input1, input2, find_table, output, sort_table;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/duplicate_data_0.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input1, read_options);

  std::string test_file2 = "/tmp/duplicate_str_data_0.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file2, input2, read_options);

  // find table
  std::string find_file = "/tmp/find_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, find_file, find_table, read_options);

  if (!status.is_ok()) {
    LOG(ERROR) << "Table Creation Failed";
  }

  std::cout << "Input Table" << std::endl;

  input1->Print();

  // Create HashIndex

  std::cout << "Create Hash HashIndex " << std::endl;

  const int index_column = 0;
  bool drop_index = true;

  std::shared_ptr<cylon::BaseIndex> index, index_str;
  std::shared_ptr<cylon::Table> indexed_table;

  cylon::IndexUtil::BuildHashIndex(index, input1, index_column);

  cylon::IndexUtil::BuildHashIndex(index_str, input2, index_column);

  LOG(INFO) << "Testing Table Properties and Functions";

  input1->Set_Index(index, drop_index);

  input2->Set_Index(index_str, drop_index);


  // Create Range HashIndex

  std::cout << "Create Range HashIndex " << std::endl;

  std::shared_ptr<cylon::RangeIndex> range_index;
  std::shared_ptr<cylon::BaseIndex> bindex;
  std::shared_ptr<cylon::Table> range_indexed_table;

  cylon::IndexUtil::BuildHashIndex(bindex, input1, 0);

  range_index = std::static_pointer_cast<cylon::RangeIndex>(bindex);

  std::cout << "Start : " << range_index->GetStart() << std::endl;
  std::cout << "Step : " << range_index->GetStep() << std::endl;
  std::cout << "Stop : " << range_index->GetAnEnd() << std::endl;

  // BaseIndexer

  std::shared_ptr<cylon::BaseIndexer> base_indexer = std::make_shared<cylon::LocIndexer>(cylon::IndexingSchema::Hash);

//  // loc mode 1
  long start_index = 4;
  long end_index = 27;
  int column = 0;
  std::shared_ptr<cylon::Table> loc_tb1;
  status = base_indexer->loc(&start_index, &end_index, column, input1, loc_tb1);

  if (!status.is_ok()) {
    return 1;
  }

  std::string statement_loc1 = "Loc 1";

  separator(statement_loc1);

  loc_tb1->Print();

  LOG(INFO) << "HashIndex of loc[1] table";
  auto loc_index_arr_1 = loc_tb1->GetIndex()->GetIndexArray();
  print_arrow_array(loc_index_arr_1);

  // loc mode 2
  long start_index_1 = 10;
  long end_index_1 = 123;
  int start_column_index = 0;
  int end_column_index = 1;
  std::shared_ptr<cylon::Table> loc_tb2;

  base_indexer->loc(&start_index_1, &end_index_1, start_column_index, end_column_index, input1, loc_tb2);

  std::string statement_loc2 = "Loc 2";

  separator(statement_loc2);

  loc_tb2->Print();

  LOG(INFO) << "HashIndex of loc[2] table";
  auto loc_index_arr_2 = loc_tb2->GetIndex()->GetIndexArray();
  print_arrow_array(loc_index_arr_2);

//  // loc mode 3
  long start_index_2 = 27;
  long end_index_2 = 123;
  std::vector<int> cols = {0, 2};
  std::shared_ptr<cylon::Table> loc_tb3;

  base_indexer->loc(&start_index_2, &end_index_2, cols, input1, loc_tb3);

  std::string statement_loc3 = "Loc 3";

  separator(statement_loc3);

  loc_tb3->Print();

  LOG(INFO) << "HashIndex of loc[3] table";
  auto loc_index_arr_3 = loc_tb3->GetIndex()->GetIndexArray();
  print_arrow_array(loc_index_arr_3);

//
//  // loc mode 4
//
  long start_index_3 = 10;
  int select_column = 1;
  std::shared_ptr<cylon::Table> loc_tb4;

  base_indexer->loc(&start_index_3, select_column, input1, loc_tb4);

  std::string statement_loc4 = "Loc 4";

  separator(statement_loc4);

  std::string x = "l";

  x = x + std::to_string(cylon::IndexingSchema::Range);

  loc_tb4->Print();

  LOG(INFO) << "HashIndex of loc[4] table";
  auto loc_index_arr_4 = loc_tb4->GetIndex()->GetIndexArray();
  print_arrow_array(loc_index_arr_4);
//
//  // loc mode 5
//
  typedef std::vector<void *> vector_void_star;

  vector_void_star output_items;

  std::vector<long> start_indices_5 = {4, 1};

  for (size_t tx = 0; tx < start_indices_5.size(); tx++) {
    output_items.push_back(reinterpret_cast<void *const>(start_indices_5.at(tx)));
  }

  int start_column_5 = 1;
  int end_column_5 = 2;
  std::shared_ptr<cylon::Table> loc_tb5;

  base_indexer->loc(output_items, start_column_5, end_column_5, input1, loc_tb5);

  std::string statement_loc5 = "Loc 5";

  separator(statement_loc5);

  loc_tb5->Print();

  LOG(INFO) << "HashIndex of loc[5] table";
  auto loc_index_arr_5 = loc_tb5->GetIndex()->GetIndexArray();
  print_arrow_array(loc_index_arr_5);

//
//  // loc mode 6
//
  vector_void_star output_items1;

  std::vector<long> start_indices_6{4, 1};

  for (size_t tx = 0; tx < start_indices_6.size(); tx++) {
    output_items1.push_back(reinterpret_cast<void *const>(start_indices_6.at(tx)));
  }
  std::vector<int> columns = {1, 2};

  std::shared_ptr<cylon::Table> loc_tb6;

  base_indexer->loc(output_items1, columns, input1, loc_tb6);

  std::string statement_loc6 = "Loc 6";

  separator(statement_loc6);

  loc_tb6->Print();

  LOG(INFO) << "HashIndex of loc[6] table";
  auto loc_index_arr_6 = loc_tb6->GetIndex()->GetIndexArray();
  print_arrow_array(loc_index_arr_6);
//
//  // loc mode 7
//
  long start_index_7 = 4;

  std::vector<int> columns_7 = {1, 2};

  std::shared_ptr<cylon::Table> loc_tb7;

  base_indexer->loc(&start_index_7, columns, input1, loc_tb7);

  std::string statement_loc7 = "Loc 7";

  separator(statement_loc7);

  loc_tb7->Print();

  LOG(INFO) << "HashIndex of loc[7] table";
  auto loc_index_arr_7 = loc_tb7->GetIndex()->GetIndexArray();
  print_arrow_array(loc_index_arr_7);
//
//
//  // loc mode 8
//
  long start_index_8 = 4;

  int start_column_8 = 1;
  int end_column_8 = 2;

  std::shared_ptr<cylon::Table> loc_tb8;

  base_indexer->loc(&start_index_8, start_column_8, end_column_8, input1, loc_tb8);

  std::string statement_loc8 = "Loc 8";

  separator(statement_loc8);

  loc_tb8->Print();

  LOG(INFO) << "HashIndex of loc[8] table";
  auto loc_index_arr_8 = loc_tb8->GetIndex()->GetIndexArray();
  print_arrow_array(loc_index_arr_8);
//
//  // loc mode 9
//
  int column_9 = 1;

  std::shared_ptr<cylon::Table> loc_tb9;

  base_indexer->loc(output_items1, column_9, input1, loc_tb9);

  std::string statement_loc9 = "Loc 9";

  separator(statement_loc9);

  loc_tb9->Print();

  LOG(INFO) << "HashIndex of loc[9] table";
  auto loc_index_arr_9 = loc_tb9->GetIndex()->GetIndexArray();
  print_arrow_array(loc_index_arr_9);
  /**
   * For string data
   * */

  //loc mode 1
  std::string s_start_index("e");
  std::string s_end_index("j");

  int s_column = 0;
  std::shared_ptr<cylon::Table> loc_s_tb1;
  base_indexer->loc(&s_start_index, &s_end_index, s_column, input2, loc_s_tb1);

  std::string statement_s_loc1 = "Loc S 1";

  separator(statement_s_loc1);

  loc_s_tb1->Print();

  std::shared_ptr<arrow::Array> arr = index_str->GetIndexAsArray();

  for (int64_t xi = 0; xi < arr->length(); xi++) {
    auto result = arr->GetScalar(xi);
    std::cout << " " << result.ValueOrDie()->ToString();
  }
  std::cout << std::endl;

  LOG(INFO) << "HashIndex of loc[str] table";
  auto loc_index_arr_str = loc_s_tb1->GetIndex()->GetIndexArray();
  print_arrow_array(loc_index_arr_str);

  build_array_from_vector(ctx);

  return 0;
}

int indexing_benchmark() {

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/indexing_10000000_0.9.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  const int index_column = 0;
  bool drop_index = true;

  std::shared_ptr<cylon::BaseIndex> index, index_str;
  std::shared_ptr<cylon::Table> indexed_table;

  std::shared_ptr<cylon::BaseIndexer> base_indexer = std::make_shared<cylon::LocIndexer>(cylon::IndexingSchema::Hash);

  auto start_start = std::chrono::steady_clock::now();

  cylon::IndexUtil::BuildLinearIndex(index, input, index_column);

  auto read_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Indexing table in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                read_end_time - start_start).count() << "[ms]";

  input->Set_Index(index, drop_index);

  long value = 237250;
  int start_column = 1;
  int end_column = 2;

  auto start_start_i = std::chrono::steady_clock::now();
  base_indexer->loc(&value, start_column, end_column, input, output);
  auto end_start_i = std::chrono::steady_clock::now();

  LOG(INFO) << "Loc table in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                end_start_i - start_start_i).count() << "[ms]";

  auto start_start_j = std::chrono::steady_clock::now();
  auto index_arr = index->GetIndexAsArray();
  auto end_start_j = std::chrono::steady_clock::now();
  LOG(INFO) << "Get HashIndex Arr table in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                end_start_j - start_start_j).count() << "[ms]";

  output->Print();

  return 0;
}

void separator(std::string &statement) {
  std::cout << "===============================================" << std::endl;
  std::cout << statement << std::endl;
  std::cout << "===============================================" << std::endl;
}

int build_int_index_from_values(std::shared_ptr<cylon::CylonContext> &ctx) {
  arrow::Status arrow_status;

  std::shared_ptr<cylon::BaseIndex> custom_index;
  std::shared_ptr<arrow::Array> index_values;
  std::vector<int32_t> ix_vals = {1, 2, 3, 4, 11};
  auto pool = cylon::ToArrowPool(ctx);
  arrow::Int32Builder int_32_builder(pool);

  int_32_builder.AppendValues(ix_vals);
  arrow_status = int_32_builder.Finish(&index_values);

  if (!arrow_status.ok()) {
    return -1;
  }

  cylon::IndexUtil::BuildHashIndexFromArray(index_values, pool, custom_index);

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

  std::shared_ptr<cylon::BaseIndex> custom_index;
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

  cylon::IndexUtil::BuildHashIndexFromArray(index_values, pool, custom_index);

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

int build_array_from_vector(std::shared_ptr<cylon::CylonContext> &ctx) {

  LOG(INFO) << "BuildHashIndex index from vector";
  std::vector<int> index_vector0{1, 2, 3, 4, 5};
  std::vector<int16_t> index_vector{1, 2, 3, 4, 5};
  std::vector<int32_t> index_vector1{1, 2, 3, 4, 5};
  std::vector<int64_t> index_vector2{1, 2, 3, 4, 5};
  std::vector<double_t> index_vector3{1, 2, 3, 4, 5};
  std::vector<float_t> index_vector4{1, 2, 3, 4, 5};

  std::shared_ptr<cylon::BaseIndex> index;
  auto pool = cylon::ToArrowPool(ctx);
  cylon::IndexUtil::BuildHashIndexFromVector(index_vector3, pool, index);

  auto arr = index->GetIndexAsArray();
  LOG(INFO) << "Array length : " << arr->length();

  std::shared_ptr<arrow::DoubleScalar> int_32_scalar;

  for (int64_t xi = 0; xi < arr->length(); xi++) {
    auto result = arr->GetScalar(xi);
    if (!result.ok()) {
      return -1;
    }
    auto scalar = result.ValueOrDie();
    int_32_scalar = std::static_pointer_cast<arrow::DoubleScalar>(scalar);
    std::cout << " " << int_32_scalar->value;
  }
  std::cout << std::endl;

  return 0;
}

int test_loc_operations(cylon::IndexingSchema schema) {
  std::string func_title = "[Numeric] Testing Indexing Schema " + std::to_string(schema);
  separator(func_title);
  LOG(INFO) << "Testing Indexing Schema " << schema;
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

  LOG(INFO) << "Original Data";

  input->Print();

  long start_index = 0;
  long end_index = 5;
  int64_t column = 0;
  int start_column = 0;
  int end_column = 1;
  std::vector<int> columns = {0, 1};
  typedef std::vector<void *> vector_void_star;

  vector_void_star output_items;

  std::vector<long> start_indices = {4, 1};

  for (size_t tx = 0; tx < start_indices.size(); tx++) {
    long *val = new long(start_indices.at(tx));
    output_items.push_back(static_cast<void *>(val));
  }
  bool drop_index = true;

  std::shared_ptr<cylon::BaseIndex> index;

  status = cylon::IndexUtil::BuildIndex(schema, input, 0, index);

  if (!status.is_ok()) {
    return -1;
  }

  status = input->Set_Index(index, drop_index);

  if (!status.is_ok()) {
    return -1;
  }

  LOG(INFO) << "[RangeIndex] Records in Table Rows: " << input->Rows() << ", Columns: " << input->Columns();

  std::shared_ptr<cylon::BaseIndexer> loc_indexer = std::make_shared<cylon::LocIndexer>(schema);

  LOG(INFO) << "LOC Mode 1 Example";

  loc_indexer->loc(&start_index, &end_index, column, input, output);

  output->Print();

  LOG(INFO) << "LOC Mode 2 Example";

  loc_indexer->loc(&start_index, &end_index, start_column, end_column, input, output);

  output->Print();

  LOG(INFO) << "LOC Mode 3 Example";

  loc_indexer->loc(&start_index, &end_index, columns, input, output);

  output->Print();

  LOG(INFO) << "LOC Mode 4 Example";

  loc_indexer->loc(&start_index, column, input, output);

  output->Print();

  LOG(INFO) << "LOC Mode 5 Example";

  loc_indexer->loc(&start_index, start_column, end_column, input, output);

  output->Print();

  LOG(INFO) << "LOC Mode 6 Example";

  loc_indexer->loc(&start_index, columns, input, output);

  output->Print();

  LOG(INFO) << "LOC Mode 7 Example";

  loc_indexer->loc(output_items, column, input, output);

  output->Print();

  LOG(INFO) << "LOC Mode 8 Example";

  loc_indexer->loc(output_items, start_column, end_column, input, output);

  output->Print();

  LOG(INFO) << "LOC Mode 9 Example";

  loc_indexer->loc(output_items, columns, input, output);

  output->Print();

  return 0;
}

int test_str_loc_operations(cylon::IndexingSchema schema) {
  std::string func_title = "[String] Testing Indexing Schema " + std::to_string(schema);
  separator(func_title);
  LOG(INFO) << "Testing Indexing Schema " << schema;
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output, output1;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/home/vibhatha/build/cylon/data/input/indexing_str_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input, read_options);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in creating table";
    return -1;
  }

  LOG(INFO) << "Original Data";

  input->Print();

  std::string start_index = "f";
  std::string end_index = "m";
  int64_t column = 0;
  int start_column = 0;
  int end_column = 1;
  std::vector<int> columns = {0, 1};
  typedef std::vector<void *> vector_void_star;

  vector_void_star output_items;

  std::vector<std::string> start_indices = {"e", "k"};
  LOG(INFO) << "Filling";
  for (size_t tx = 0; tx < start_indices.size(); tx++) {
    std::string *sval = new std::string(start_indices.at(tx));
    std::cout << *sval << ":" << sval << std::endl;
    output_items.push_back(static_cast<void *>(sval));
  }
  LOG(INFO) << "Loading";
  for (size_t i = 0; i < output_items.size(); i++) {
    void *pStr = output_items.at(i);
    std::string &s = *(static_cast<std::string *>(pStr));
    std::cout << s << ":" << &s << std::endl;
  }
  bool drop_index = true;

  std::shared_ptr<cylon::BaseIndex> index;

  status = cylon::IndexUtil::BuildIndex(schema, input, 0, index);

  if (!status.is_ok()) {
    return -1;
  }

  status = input->Set_Index(index, drop_index);

  if (!status.is_ok()) {
    return -1;
  }

  LOG(INFO) << "[RangeIndex] Records in Table Rows: " << input->Rows() << ", Columns: " << input->Columns();

  std::shared_ptr<cylon::BaseIndexer> loc_indexer = std::make_shared<cylon::LocIndexer>(schema);

  LOG(INFO) << "LOC Mode 1 Example";

  loc_indexer->loc(&start_index, &end_index, column, input, output);

  output->Print();

  LOG(INFO) << "LOC Mode 2 Example";

  loc_indexer->loc(&start_index, &end_index, start_column, end_column, input, output);

  output->Print();

  LOG(INFO) << "LOC Mode 3 Example";

  loc_indexer->loc(&start_index, &end_index, columns, input, output);

  output->Print();

  LOG(INFO) << "LOC Mode 4 Example";

  loc_indexer->loc(&start_index, column, input, output);

  output->Print();

  LOG(INFO) << "LOC Mode 5 Example";

  loc_indexer->loc(&start_index, start_column, end_column, input, output);

  output->Print();

  LOG(INFO) << "LOC Mode 6 Example";

  loc_indexer->loc(&start_index, columns, input, output);

  output->Print();

  LOG(INFO) << "LOC Mode 7 Example";

  status = loc_indexer->loc(output_items, column, input, output);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in operation LOC Mode 7";
  }

  output->Print();

  LOG(INFO) << "LOC Mode 8 Example";

  status = loc_indexer->loc(output_items, start_column, end_column, input, output);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in operation LOC Mode 8";
  }

  output->Print();

  LOG(INFO) << "LOC Mode 9 Example";

  status = loc_indexer->loc(output_items, columns, input, output);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in operation LOC Mode 9";
  }

  output->Print();

  return 0;
}

int test_iloc_operations() {
  std::string func_title = "Testing ILoc ";
  separator(func_title);
  LOG(INFO) << func_title;
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

  LOG(INFO) << "Original Data";

  input->Print();

  long start_index = 0;
  long end_index = 5;
  int64_t column = 0;
  int start_column = 0;
  int end_column = 1;
  std::vector<int> columns = {0, 1};
  typedef std::vector<void *> vector_void_star;

  vector_void_star output_items;

  std::vector<long> start_indices = {4, 1};

  for (size_t tx = 0; tx < start_indices.size(); tx++) {
    output_items.push_back(reinterpret_cast<void *const>(start_indices.at(tx)));
  }
  bool drop_index = true;

  std::shared_ptr<cylon::BaseIndex> index, loc_index, range_index;

  status = cylon::IndexUtil::BuildIndex(cylon::IndexingSchema::Range, input, 0, range_index);

  if (!status.is_ok()) {
    return -1;
  }

  status = input->Set_Index(range_index, drop_index);

  if (!status.is_ok()) {
    return -1;
  }

  LOG(INFO) << "[RangeIndex] Records in Table Rows: " << input->Rows() << ", Columns: " << input->Columns();

  std::shared_ptr<cylon::BaseIndexer> loc_indexer = std::make_shared<cylon::ILocIndexer>(cylon::IndexingSchema::Range);

  LOG(INFO) << "iLOC Mode 1 Example";

  loc_indexer->loc(&start_index, &end_index, column, input, output);

  output->Print();

  LOG(INFO) << "iLOC Mode 2 Example";

  loc_indexer->loc(&start_index, &end_index, start_column, end_column, input, output);

  output->Print();

  LOG(INFO) << "iLOC Mode 3 Example";

  loc_indexer->loc(&start_index, &end_index, columns, input, output);

  output->Print();

  LOG(INFO) << "iLOC Mode 4 Example";

  loc_indexer->loc(&start_index, column, input, output);

  output->Print();

  LOG(INFO) << "iLOC Mode 5 Example";

  loc_indexer->loc(&start_index, start_column, end_column, input, output);

  output->Print();

  LOG(INFO) << "iLOC Mode 6 Example";

  loc_indexer->loc(&start_index, columns, input, output);

  output->Print();

  LOG(INFO) << "iLOC Mode 7 Example";

  loc_indexer->loc(output_items, column, input, output);

  output->Print();

  LOG(INFO) << "iLOC Mode 8 Example";

  loc_indexer->loc(output_items, start_column, end_column, input, output);

  output->Print();

  LOG(INFO) << "iLOC Mode 9 Example";

  loc_indexer->loc(output_items, columns, input, output);

  output->Print();

  return 0;
}

int print_arrow_array(std::shared_ptr<arrow::Array> &arr) {
  for (int64_t xi = 0; xi < arr->length(); xi++) {
    auto result = arr->GetScalar(xi);
    std::cout << " " << result.ValueOrDie()->ToString();
  }
  std::cout << std::endl;
  return 0;
}


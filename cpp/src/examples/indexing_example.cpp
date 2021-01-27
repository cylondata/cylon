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

int test_loc_operations(cylon::IndexingSchema schema);

int test_str_loc_operations(cylon::IndexingSchema schema);

int print_arrow_array(std::shared_ptr<arrow::Array> &arr);

int print_index_output(std::shared_ptr<cylon::Table> &output, std::string &message);


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

  std::vector<cylon::IndexingSchema> schemas{cylon::IndexingSchema::Range,
                                             cylon::IndexingSchema::Linear,
                                             cylon::IndexingSchema::Hash};
  for (auto schema : schemas) {
    test_loc_operations(schema);
  }

  for (size_t i = 1; i < schemas.size() ; i++) {
    test_str_loc_operations(schemas.at(i));
  }

  test_iloc_operations();

  std::cout << "Data Type : " << arrow::Int32Type::CTypeImpl::type_id << std::endl;

  long value = 5;
  void * ptr = static_cast<void*> (&value);
  long * cast_value = static_cast<long*>(ptr);
  std::cout << "Value : " << ptr << ", " <<  *cast_value << std::endl;

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
  std::shared_ptr<arrow::Array> index_arr;

  std::shared_ptr<cylon::BaseIndex> index;

  status = cylon::IndexUtil::BuildIndex(schema, input, 0, index);

  if (!status.is_ok()) {
    return -1;
  }

  status = input->Set_Index(index, drop_index);

  if (!status.is_ok()) {
    return -1;
  }

  std::string loc_output_msg;

  LOG(INFO) << "[RangeIndex] Records in Table Rows: " << input->Rows() << ", Columns: " << input->Columns();

  std::shared_ptr<cylon::BaseIndexer> loc_indexer = std::make_shared<cylon::LocIndexer>(schema);

  LOG(INFO) << "LOC Mode 1 Example";

  loc_indexer->loc(ptr, &end_index, column, input, output);

  loc_output_msg = " LOC 1 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "LOC Mode 2 Example";

  loc_indexer->loc(&start_index, &end_index, start_column, end_column, input, output);

  loc_output_msg = " LOC 2 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "LOC Mode 3 Example";

  loc_indexer->loc(&start_index, &end_index, columns, input, output);

  loc_output_msg = " LOC 3 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "LOC Mode 4 Example";

  loc_indexer->loc(&start_index, column, input, output);

  loc_output_msg = " LOC 4 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "LOC Mode 5 Example";

  loc_indexer->loc(&start_index, start_column, end_column, input, output);

  loc_output_msg = " LOC 5 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "LOC Mode 6 Example";

  loc_indexer->loc(&start_index, columns, input, output);

  loc_output_msg = " LOC 6 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "LOC Mode 7 Example";

  loc_indexer->loc(output_items, column, input, output);

  loc_output_msg = " LOC 7 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "LOC Mode 8 Example";

  loc_indexer->loc(output_items, start_column, end_column, input, output);

  loc_output_msg = " LOC 8 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "LOC Mode 9 Example";

  loc_indexer->loc(output_items, columns, input, output);

  loc_output_msg = " LOC 9 ";
  print_index_output(output, loc_output_msg);

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

  std::string loc_output_msg;

  LOG(INFO) << "[RangeIndex] Records in Table Rows: " << input->Rows() << ", Columns: " << input->Columns();

  std::shared_ptr<cylon::BaseIndexer> loc_indexer = std::make_shared<cylon::LocIndexer>(schema);

  LOG(INFO) << "LOC Mode 1 Example";

  loc_indexer->loc(&start_index, &end_index, column, input, output);

  loc_output_msg = " LOC 1 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "LOC Mode 2 Example";

  loc_indexer->loc(&start_index, &end_index, start_column, end_column, input, output);

  loc_output_msg = " LOC 2 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "LOC Mode 3 Example";

  loc_indexer->loc(&start_index, &end_index, columns, input, output);

  loc_output_msg = " LOC 3 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "LOC Mode 4 Example";

  loc_indexer->loc(&start_index, column, input, output);

  loc_output_msg = " LOC 4 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "LOC Mode 5 Example";

  loc_indexer->loc(&start_index, start_column, end_column, input, output);

  loc_output_msg = " LOC 5 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "LOC Mode 6 Example";

  loc_indexer->loc(&start_index, columns, input, output);

  loc_output_msg = " LOC 6 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "LOC Mode 7 Example";

  status = loc_indexer->loc(output_items, column, input, output);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in operation LOC Mode 7";
  }

  loc_output_msg = " LOC 7 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "LOC Mode 8 Example";

  status = loc_indexer->loc(output_items, start_column, end_column, input, output);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in operation LOC Mode 8";
  }

  loc_output_msg = " LOC 8 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "LOC Mode 9 Example";

  status = loc_indexer->loc(output_items, columns, input, output);

  if (!status.is_ok()) {
    LOG(ERROR) << "Error occurred in operation LOC Mode 9";
  }

  loc_output_msg = " LOC 9 ";
  print_index_output(output, loc_output_msg);

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
    long *val = new long(start_indices.at(tx));
    output_items.push_back(static_cast<void *>(val));
  }
  bool drop_index = true;
  std::string loc_output_msg;

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

  loc_output_msg = " iLOC 1 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "iLOC Mode 2 Example";

  loc_indexer->loc(&start_index, &end_index, start_column, end_column, input, output);

  loc_output_msg = " iLOC 1 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "iLOC Mode 3 Example";

  loc_indexer->loc(&start_index, &end_index, columns, input, output);

  loc_output_msg = " iLOC 1 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "iLOC Mode 4 Example";

  loc_indexer->loc(&start_index, column, input, output);

  loc_output_msg = " iLOC 1 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "iLOC Mode 5 Example";

  loc_indexer->loc(&start_index, start_column, end_column, input, output);

  loc_output_msg = " iLOC 1 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "iLOC Mode 6 Example";

  loc_indexer->loc(&start_index, columns, input, output);

  loc_output_msg = " iLOC 1 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "iLOC Mode 7 Example";

  loc_indexer->loc(output_items, column, input, output);

  loc_output_msg = " iLOC 1 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "iLOC Mode 8 Example";

  loc_indexer->loc(output_items, start_column, end_column, input, output);

  loc_output_msg = " iLOC 1 ";
  print_index_output(output, loc_output_msg);

  LOG(INFO) << "iLOC Mode 9 Example";

  loc_indexer->loc(output_items, columns, input, output);

  loc_output_msg = " iLOC 1 ";
  print_index_output(output, loc_output_msg);

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

int print_index_output(std::shared_ptr<cylon::Table> &output, std::string &message) {
  LOG(INFO) << message;
  LOG(INFO) << "Loc operation Table";
  output->Print();
  LOG(INFO) << "Resultant Index";
  auto index_arr = output->GetIndex()->GetIndexArray();
  print_arrow_array(index_arr);
  return 0;
}
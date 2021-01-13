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

#include "indexing/index_utils.hpp"

class Hasher {
 public:
  size_t operator()(std::string const &key) const {     // the parameter type should be the same as the type of key of unordered_map
    size_t hash = 0;
    for (size_t i = 0; i < key.size(); i++) {
      hash += key[i] % 7;
    }
    return hash;
  }
};

int arrow_take_test();

int vector_populate();

int indexing_simple_example();
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
  indexing_simple_example();
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
    std::cout << "Error occured in Find" << std::endl;
  } else {
    std::cout << "Find Succeeded" << std::endl;
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

  std::shared_ptr<cylon::Table> input1, find_table, output, sort_table;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/duplicate_data_0.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input1, read_options);

  // find table
  std::string find_file = "/tmp/find_data.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, find_file, find_table, read_options);

  if (!status.is_ok()) {
    LOG(ERROR) << "Table Creation Failed";
  }

  std::cout << "Input Table" << std::endl;

  input1->Print();

  const int index_column = 0;
  bool drop_index = true;

  //
//  input1->Set_Index(index_column, drop_index, output);
//
//  long search_val = 4;
//
//  std::cout << "Find Result" << std::endl;
//
//  input1->Find_From_Index(&search_val, index_column, output);
//
//  std::cout << "============================" << std::endl;
//
//  output->Print();
//
//  std::cout << "============================" << std::endl;
//
//  //arrow_take_test(ctx, input1);
//
//  std::shared_ptr<arrow::Array> arr = input1->GetIndex()->GetIndex();
//  std::cout << "Index Values" << std::endl;
//
//  for (int64_t i = 0; i < arr->length(); i++) {
//    std::cout << arr->GetScalar(i).ValueOrDie()->ToString() << std::endl;
//  }

    std::shared_ptr<cylon::BaseIndex> index;
    std::shared_ptr<cylon::Table> indexed_table;

    long search_value = 4;

    cylon::IndexUtil::Build(index, input1, index_column, drop_index, indexed_table);

    LOG(INFO) << "Testing IndexUtils ";
    find_table->Print();

    LOG(INFO) << "Testing Table Properties and Functions";

    input1->Set_Index(index, drop_index);

    input1->Find(&search_value, find_table);

    find_table->Print();




  return 0;
}

int vector_populate() {
  auto start_start = std::chrono::steady_clock::now();
  int size = 10;
  std::vector<int64_t> vec(size, 1);
  auto end_start = std::chrono::steady_clock::now();

  LOG(INFO) << "Vector Generation time "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                end_start - start_start).count() << "[ms]";

  auto p1 = std::chrono::steady_clock::now();
  for (int idx = 0; idx < vec.size(); idx++) {
    vec[idx] = idx;
  }
  auto p2 = std::chrono::steady_clock::now();

  LOG(INFO) << "Vector Repopulate time "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                p2 - p1).count() << "[ms]";

  for (int idx = 0; idx < vec.size(); idx++) {
    std::cout << vec[idx] << " ";
  }
  std::cout << std::endl;


  return 0;
}




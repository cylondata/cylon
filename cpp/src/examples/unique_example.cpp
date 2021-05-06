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

int sequential(std::shared_ptr<cylon::Table> &table, std::shared_ptr<cylon::Table> &out, const std::vector<int> &cols);
int distributed(std::shared_ptr<cylon::Table> &table, std::shared_ptr<cylon::Table> &out, const std::vector<int> &cols);
int dummy_test();
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

int do_unique(int argc, char *argv[]);

int main(int argc, char *argv[]) {
  do_unique(argc, argv);
  std::cout << "========================" << std::endl;
  dummy_test();
}

int sequential(std::shared_ptr<cylon::Table> &table, std::shared_ptr<cylon::Table> &out, const std::vector<int> &cols) {
  // apply unique operation
  const auto& ctx = table->GetContext();
  auto status = cylon::Unique(table, cols, out, true);
  if (!status.is_ok()) {
    LOG(INFO) << "Unique failed " << status.get_msg();
    ctx->Finalize();
    return 1;
  }
  return 0;
}

int distributed(std::shared_ptr<cylon::Table> &table,
                std::shared_ptr<cylon::Table> &out,
                const std::vector<int> &cols) {
  const auto& ctx = table->GetContext();
  auto status = cylon::DistributedUnique(table, cols, out);
  if (!status.is_ok()) {
    LOG(INFO) << "Distributed Unique failed " << status.get_msg();
    ctx->Finalize();
    return 1;
  }
  auto write_opts = cylon::io::config::CSVWriteOptions().WithDelimiter(',');
  cylon::WriteCSV(out, "/tmp/dist_unique_" + std::to_string(ctx->GetRank()), write_opts);
  return 0;
}

int dummy_test() {
  std::cout << "Dummy Test" << std::endl;
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input1, output, sort_table;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::string test_file = "/tmp/duplicate_data_0.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input1, read_options);

  if (!status.is_ok()) {
    LOG(ERROR) << "Table Creation Failed";
  }

  std::cout << "Input Table" << std::endl;
  input1->Print();

  std::vector<int> cols = {0};
  cylon::Unique(input1, cols, output, true);

  cylon::Sort(output, 3, sort_table);

  std::cout << "Output Table" << std::endl;
  sort_table->Print();

  LOG(INFO) << "First table had : " << input1->Rows()
            << ", Unique has : "
            << sort_table->Rows() << " rows";

  std::shared_ptr<arrow::Table> artb;
  sort_table->ToArrowTable(artb);

  std::vector<int32_t> outval3 = {1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15};
  int count = 0;

  const std::shared_ptr<arrow::Int64Array>
      &carr = std::static_pointer_cast<arrow::Int64Array>(artb->column(3)->chunk(0));
  for (int i = 0; i < carr->length(); i++) {
    std::cout << carr->Value(i) << std::endl;
    if (carr->Value(i) == outval3.at(i)) {
      count++;
    }
  }

  return 0;
}

int do_unique(int argc, char *argv[]) {
  if (argc < 2) {
    LOG(ERROR) << "There should be 2 args. count, duplication factor";
    return 1;
  }

  auto start_time = std::chrono::steady_clock::now();
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  std::shared_ptr<cylon::Table> first_table, unique_table, sorted_table;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << argv[1] << std::endl;
  auto status = cylon::FromCSV(ctx, argv[1], first_table, read_options);
  if (!status.is_ok()) {
    LOG(INFO) << "Table reading failed " << argv[1];
    ctx->Finalize();
    return 1;
  }

  auto read_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Read all in " << std::chrono::duration_cast<std::chrono::milliseconds>(
      read_end_time - start_time).count() << "[ms]";

  auto union_start_time = std::chrono::steady_clock::now();
  std::vector<int> cols = {0};

  if (ctx->GetWorldSize() == 1) {
    sequential(first_table, unique_table, cols);
  } else {
    distributed(first_table, unique_table, cols);
  }

  cylon::Sort(unique_table, 3, sorted_table);

  read_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "First table had : " << first_table->Rows()
            << ", Unique has : "
            << sorted_table->Rows() << " rows";
  LOG(INFO) << "Unique done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                read_end_time - union_start_time).count()
            << "[ms]";

  auto end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "Operation took : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count() << "[ms]";

  std::cout << " Original Data" << std::endl;

  first_table->Print();

  std::cout << " Unique Data" << std::endl;

  sorted_table->Print();

  //ctx->Finalize();

  return 0;
}
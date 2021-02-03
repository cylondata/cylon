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

int run_indexing_benchmark(cylon::IndexingSchema schema);

int main(int argc, char *argv[]) {

  std::string file_path = argv[1];
  long search_val = atol(argv[2]);
  std::vector<cylon::IndexingSchema> schemas{cylon::IndexingSchema::Range,
                                             cylon::IndexingSchema::Linear,
                                             cylon::IndexingSchema::Hash};

  for(auto schema : schemas) {
    run_indexing_benchmark(schema, search_val);
  }

  return 0;
}

int run_indexing_benchmark(cylon::IndexingSchema schema, std::string &file_path, long &search_val) {

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  cylon::Status status;

  std::shared_ptr<cylon::Table> input, output;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  // read first table

  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, file_path, input, read_options);

  const int index_column = 0;
  bool drop_index = true;

  std::shared_ptr<cylon::BaseIndex> index, index_str;
  std::shared_ptr<cylon::Table> indexed_table;

  std::shared_ptr<cylon::BaseIndexer> base_indexer = std::make_shared<cylon::LocIndexer>(schema);

  auto start_start = std::chrono::steady_clock::now();

  cylon::IndexUtil::BuildIndex(schema, index, input, index_column);

  auto read_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Indexing table in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                read_end_time - start_start).count() << "[ms]";

  input->Set_Index(index, drop_index);

  int start_column = 1;
  int end_column = 2;

  auto start_start_i = std::chrono::steady_clock::now();
  base_indexer->loc(&search_val, start_column, end_column, input, output);
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
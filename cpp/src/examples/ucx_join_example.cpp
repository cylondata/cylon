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

#include <net/ucx/ucx_communicator.hpp>
#include <ctx/cylon_context.hpp>
#include <table.hpp>

int main(int argc, char *argv[]) {
  // TODO Sandeepa remove this
//  char* tst = "Hellloo";
//  void* pnt = (void *)tst;
//  std::cout << pnt << " | " << (char*)pnt << std::endl;
  std::cout << "UCX Process started" << std::endl;
//  int i = 0;
//  while(0==i)
//    sleep(5);

  if (argc < 3) {
    LOG(ERROR) << "There should be two arguments with paths to csv files";
    return 1;
  }

  auto start_start = std::chrono::steady_clock::now();
  auto mpi_config = std::make_shared<cylon::net::UCXConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  std::shared_ptr<cylon::Table> first_table, second_table, joined;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);
  auto status = cylon::FromCSV(ctx, argv[1], first_table, read_options);
  if (!status.is_ok()) {
    LOG(INFO) << "Table reading failed " << argv[1];
    ctx->Finalize();
    return 1;
  }

  status = cylon::FromCSV(ctx, argv[2], second_table, read_options);
  if (!status.is_ok()) {
    LOG(INFO) << "Table reading failed " << argv[2];
    ctx->Finalize();
    return 1;
  }
  auto read_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "Read tables in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                read_end_time - start_start).count() << "[ms]";

  status = cylon::DistributedJoin(first_table, second_table,
                              cylon::join::config::JoinConfig::InnerJoin(0, 0), joined);

  if (!status.is_ok()) {
    LOG(INFO) << "Table join failed ";
    ctx->Finalize();
    return 1;
  }
  auto join_end_time = std::chrono::steady_clock::now();
  std::shared_ptr<arrow::Table> arr_table;

  LOG(INFO) << "First table had : " << first_table->Rows() << " and Second table had : "
            << second_table->Rows() << ", Joined has : " << joined->Rows();
  LOG(INFO) << "Join done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                join_end_time - read_end_time).count() << "[ms]";

  joined->ToArrowTable(arr_table);
//  LOG(INFO) <<  arr_table->ToString();

  ctx->Finalize();
  return 0;
}

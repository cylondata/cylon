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
#include <groupby/groupby.hpp>
#include <groupby/hash_groupby.hpp>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    LOG(ERROR) << "There should be two arguments with paths to csv files";
    return 1;
  }

  auto start_start = std::chrono::steady_clock::now();
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  std::shared_ptr<cylon::Table> first_table, output;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);
  auto status = cylon::FromCSV(ctx, argv[1], first_table, read_options);
  if (!status.is_ok()) {
    LOG(INFO) << "Table reading failed " << argv[1];
    ctx->Finalize();
    return 1;
  }

  auto read_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "Read tables in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(read_end_time - start_start).count() << "[ms]";

  LOG(INFO) << "Table Data";

  first_table->Print();

  std::cout << "-----------------------" << std::endl;
//  cylon::Status s = cylon::HashGroupBy(first_table, {0, 1}, {{2, cylon::compute::VAR}}, output);
//  cylon::Status s = cylon::HashGroupBy(first_table, {0, 1}, {{2, cylon::compute::VarOp::Make()}}, output);
//  cylon::Status s = cylon::DistributedHashGroupBy(first_table, {0, 1}, {2}, {cylon::compute::VAR}, output);
//  cylon::Status s = cylon::HashGroupBy(first_table, {0, 1}, {{2, cylon::compute::NUNIQUE}}, output);
  cylon::Status s = cylon::HashGroupBy(first_table, {0, 1}, {{2, cylon::compute::QuantileOp::Make(0.2)}}, output);
//  status = cylon::DistributedHashGroupBy(first_table, {0, 1}, {2}, {cylon::compute::COUNT}, output);

  if (!status.is_ok()) {
    LOG(INFO) << "Table GroupBy failed " << status.get_msg();
    ctx->Finalize();
    return 1;
  }
  auto join_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "table had : " << first_table->Rows() << ", group_by has : " << output->Rows();
  LOG(INFO) << "group_by done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                join_end_time - read_end_time).count() << "[ms]";
  LOG(INFO) << "Output of GroupBy Operation";

  output->Print();

  LOG(INFO) << "schema" << '\n' << output->get_table()->schema()->ToString();

  ctx->Finalize();
  return 0;
}

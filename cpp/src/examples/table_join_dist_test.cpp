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

#include <net/mpi/mpi_communicator.h>
#include <ctx/twisterx_context.h>
#include <table.hpp>
#include <status.hpp>
#include <iostream>
#include <io/csv_read_config.h>
#include <chrono>

using namespace twisterx;
using namespace twisterx::join::config;

//template <const char* jtype>
bool RunJoin(int rank,
             twisterx::TwisterXContext *ctx,
             const JoinConfig &jc,
             const std::shared_ptr<Table> &table1,
             const std::shared_ptr<Table> &table2,
             std::shared_ptr<Table> &output,
             const string &h_out_path) {
  Status status;

  auto t1 = std::chrono::high_resolution_clock::now();
  status = table1->DistributedJoin(table2, jc, &output);
  auto t2 = std::chrono::high_resolution_clock::now();
  ctx->GetCommunicator()->Barrier(); // todo: should we take this inside the dist join?
  auto t3 = std::chrono::high_resolution_clock::now();

  if (!status.is_ok()) {
    LOG(ERROR) << "Join failed!";
    return false;
  }
//    status = output->WriteCSV(h_out_path);
//  auto t4 = std::chrono::high_resolution_clock::now();

  if (status.is_ok()) {
    LOG(INFO) << rank << " j_t " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " w_t " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
              << " lines " << output->Rows()
              << " t " << jc.GetType()
              << " a " << jc.GetAlgorithm();
    output->Clear();
    return true;
  } else {
    LOG(ERROR) << "Join write failed!";
    output->Clear();
    return false;
  }
}

int main(int argc, char *argv[]) {

  std::shared_ptr<Table> table1, table2, joined;
  Status status;

  auto mpi_config = new twisterx::net::MPIConfig();
  auto ctx = twisterx::TwisterXContext::InitDistributed(mpi_config);

  int rank = ctx->GetRank();
  std::string srank = std::to_string(rank);

  if (argc != 3) {
    LOG(ERROR) << "src_dir and base_dir not provided! ";
    return 1;
  }

  std::string src_dir = argv[1];
  std::string base_dir = argv[2];

  system(("mkdir -p " + base_dir).c_str());

  std::string csv1 = base_dir + "/csv1_" + srank + ".csv";
  std::string csv2 = base_dir + "/csv2_" + srank + ".csv";

  system(("cp " + src_dir + "/csv1_" + srank + ".csv " + csv1).c_str());
  system(("cp " + src_dir + "/csv2_" + srank + ".csv " + csv2).c_str());

  LOG(INFO) << rank << " Reading tables";
  auto read_options = twisterx::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);
  if (!(status = Table::FromCSV(ctx, csv1, table1, read_options)).is_ok()) {
    LOG(ERROR) << "File read failed! " << csv1;
    return 1;
  }
  if (!(status = Table::FromCSV(ctx, csv2, table2, read_options)).is_ok()) {
    LOG(ERROR) << "File read failed! " << csv2;
    return 1;
  }
  ctx->GetCommunicator()->Barrier();
  LOG(INFO) << rank << " Done reading tables. rows " << table1->Rows() << " " << table2->Rows();

//  LOG(INFO) << rank << " right join start";
//  auto right_jc = JoinConfig::RightJoin(0, 0, JoinAlgorithm::HASH);
//  RunJoin(rank, ctx, right_jc, table1, table2, joined, "/scratch/dnperera/h_out_right_" + srank + ".csv");
//  auto right_jc2 = JoinConfig::RightJoin(0, 0, JoinAlgorithm::SORT);
//  RunJoin(rank, ctx, right_jc2, table1, table2, joined, "/scratch/dnperera/s_out_right_" + srank + ".csv");
//  LOG(INFO) << rank << " right join end ----------------------------------";
//
//  LOG(INFO) << rank << " left join start";
//  auto left_jc = JoinConfig::LeftJoin(0, 0, JoinAlgorithm::HASH);
//  RunJoin(rank, ctx, left_jc, table1, table2, joined, "/scratch/dnperera/h_out_left_" + srank + ".csv");
//  auto left_jc2 = JoinConfig::LeftJoin(0, 0, JoinAlgorithm::SORT);
//  RunJoin(rank, ctx, left_jc2, table1, table2, joined, base_dir + "/s_out_left_" + srank + ".csv");
//  LOG(INFO) << rank << " left join end ----------------------------------";

  LOG(INFO) << rank << " inner join start";
  auto inner_jc = JoinConfig::InnerJoin(0, 0, JoinAlgorithm::HASH);
  RunJoin(rank, ctx, inner_jc, table1, table2, joined, "/scratch/dnperera/h_out_inner_" + srank + ".csv");
  auto inner_jc2 = JoinConfig::InnerJoin(0, 0, JoinAlgorithm::SORT);
  RunJoin(rank, ctx, inner_jc2, table1, table2, joined, "/scratch/dnperera/s_out_inner_" + srank + ".csv");
  LOG(INFO) << rank << " inner join end ----------------------------------";

//  LOG(INFO) << rank << " outer join start";
//  auto outer_jc = JoinConfig::FullOuterJoin(0, 0, JoinAlgorithm::HASH);
//  RunJoin(rank, ctx, outer_jc, table1, table2, joined, "/scratch/dnperera/h_out_outer_" + srank + ".csv");
//  auto outer_jc2 = JoinConfig::FullOuterJoin(0, 0, JoinAlgorithm::SORT);
//  RunJoin(rank, ctx, outer_jc2, table1, table2, joined, "/scratch/dnperera/s_out_outer_" + srank + ".csv");
//  LOG(INFO) << rank << " outer join end ----------------------------------";

  ctx->Finalize();

  system(("rm " + csv1).c_str());
  system(("rm " + csv2).c_str());

  return 0;
}



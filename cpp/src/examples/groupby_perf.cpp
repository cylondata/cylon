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

#include <cylon/mapreduce/mapreduce.hpp>
#include <cylon/groupby/groupby.hpp>

#include "example_utils.hpp"

#define CYLON_LOG_HELP() \
  do{                    \
    LOG(ERROR) << "input arg error " << std::endl \
               << "./groupby_perf m num_tuples_per_worker dup_factor[0.0-1.0]" << std::endl \
               << "./groupby_perf f csv_file1" << std::endl; \
    return 1;                                                  \
  } while(0)

int main(int argc, char *argv[]) {
  if (argc != 3 && argc != 4) {
    CYLON_LOG_HELP();
  }

  auto start_start = std::chrono::steady_clock::now();
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  std::shared_ptr<cylon::CylonContext> ctx;
  if (!cylon::CylonContext::InitDistributed(mpi_config, &ctx).is_ok()) {
    std::cerr << "ctx init failed! " << std::endl;
    return 1;
  }

  std::shared_ptr<cylon::Table> table, output;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  std::string mem = std::string(argv[1]);
  if (mem == "m" && argc == 4) {
    LOG(INFO) << "using in-mem tables";
    int64_t count = std::stoll(argv[2]);
    double dup = std::stod(argv[3]);
    if (cylon::examples::create_in_memory_tables(count, dup, ctx, table, 0)) {
      LOG(ERROR) << "table creation failed!";
      return 1;
    }
  } else if (mem == "f" && argc == 3) {
    LOG(INFO) << "using files";
    LOG(INFO) << "loading: " << std::string(argv[2]) + std::to_string(ctx->GetRank()) + ".csv";
    if (!cylon::FromCSV(ctx, std::string(argv[2]) + std::to_string(ctx->GetRank()) + ".csv", table)
        .is_ok()) {
      LOG(ERROR) << "file reading failed!";
      return 1;
    }
  } else {
    CYLON_LOG_HELP();
  }

  ctx->Barrier();
  auto read_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Input tables created in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                read_end_time - start_start).count() << "[ms]";

  auto status = cylon::DistributedHashGroupBy(table, 0, {1, 1, 1},
                                              {cylon::compute::SUM, cylon::compute::MEAN,
                                               cylon::compute::STDDEV},
                                              output);
  auto end_time = std::chrono::steady_clock::now();

  if (!status.is_ok()) {
    LOG(INFO) << "Unique failed " << status.get_msg();
    ctx->Finalize();
    return 1;
  }

  LOG(INFO) << "Table had : " << table->Rows() << ", output has : " << output->Rows();
  LOG(INFO) << "Completed in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - read_end_time)
                .count() << "[ms]";

  read_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Input tables created in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                read_end_time - start_start).count() << "[ms]";

  cylon::mapred::AggOpVector ops{{1, cylon::compute::SumOp::Make()},
                                 {1, cylon::compute::MeanOp::Make()},
                                 {1, cylon::compute::StdDevOp::Make()}};
  status = cylon::mapred::MapredHashGroupBy(table, {0}, ops, &output);
  end_time = std::chrono::steady_clock::now();

  if (!status.is_ok()) {
    LOG(INFO) << "Unique failed " << status.get_msg();
    ctx->Finalize();
    return 1;
  }

  LOG(INFO) << "Table had : " << table->Rows() << ", output has : " << output->Rows();
  LOG(INFO) << "Completed in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - read_end_time)
                .count() << "[ms]";

  ctx->Finalize();
  return 0;
}

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
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/net/mpi/mpi_communicator.hpp>
#include <cylon/window/window.hpp>
#include <cylon/window/hash_window.hpp>
#include <cylon/table.hpp>

#define CHECK_STATUS_AND_PRINT(first_table, status, output) \
  if (!status.is_ok()) { \
      LOG(INFO) << "Table GroupBy failed " << status.get_msg(); \
      return 1; \
  }; \
  LOG(INFO) << "table had : " << first_table->Rows() << ", window has : " << output->Rows(); \
  LOG(INFO) << "Output of Window Operation";     \
  std::cout << output->get_table()->schema()->ToString() << std::endl; \
  output->Print();                                \
  std::cout << "-----------------------" << std::endl; \



int main(int argc, char *argv[]) {
  if (argc < 2) {
    LOG(ERROR) << "There should be two arguments with paths to csv files";
    return 1;
  }

  auto start_start = std::chrono::steady_clock::now();
  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  std::shared_ptr<cylon::CylonContext> ctx;
  if (!cylon::CylonContext::InitDistributed(mpi_config, &ctx).is_ok()) {
    std::cerr << "ctx init failed! " << std::endl;
    return 1;
  }

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

  //call windowing operations

  auto window_config_rolling_obs = cylon::windowing::config::WindowConfig::DefaultWithObservations(cylon::windowing::config::WindowOperation::ROLLING, 2);


  CHECK_STATUS_AND_PRINT(first_table,
                         cylon::windowing::HashWindow(window_config_rolling_obs, first_table, {0, 1},
                                            {{2, cylon::compute::NUNIQUE}}, output),
                         output);


  return 0;
}

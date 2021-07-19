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

#include <cylon/net/mpi/mpi_communicator.hpp>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/table.hpp>

#define CHECK_STATUS(status, msg) \
  if (!status.is_ok()) {          \
    LOG(ERROR) << msg << " " << status.get_msg(); \
    ctx->Finalize();              \
    return 1;                     \
  }                               

int main() {

  auto mpi_config = cylon::net::MPIConfig::Make();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
  const int rank = ctx->GetRank() + 1;

  const std::string csv1 = "/tmp/user_device_tm_" + std::to_string(rank) + ".csv";
  const std::string csv2 = "/tmp/user_usage_tm_" + std::to_string(rank) + ".csv";

  std::shared_ptr<cylon::Table> first_table, second_table, joined_table;
  cylon::Status status;

  status = cylon::FromCSV(ctx, csv1, first_table);
  CHECK_STATUS(status, "Reading csv1 failed!")

  status = cylon::FromCSV(ctx, csv2, second_table);
  CHECK_STATUS(status, "Reading csv2 failed!")

  auto join_config = cylon::join::config::JoinConfig::InnerJoin(0, 3);
  status = cylon::DistributedJoin(first_table, second_table, join_config, joined_table);
  CHECK_STATUS(status, "Join failed!")

  LOG(INFO) << "First table had : " << first_table->Rows() << " and Second table had : "
            << second_table->Rows() << ", Joined has : " << joined_table->Rows();

  ctx->Finalize();
  return 0;
}
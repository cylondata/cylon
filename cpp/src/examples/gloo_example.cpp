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

#include <iostream>

#include "cylon/net/gloo/gloo_communicator.hpp"
#include "cylon/ctx/cylon_context.hpp"

#include "examples/example_utils.hpp"

static constexpr int kCount = 10;
static constexpr double kDup = 0.9;

int main(int argc, char **argv) {
  auto config = std::make_shared<cylon::net::GlooConfig>(std::stoi(argv[1]), std::stoi(argv[2]));
  config->SetFileStorePath("/tmp/gloo/");
  config->SetStorePrefix("foo");

  if (config->rank() == 0)  {
    system("rm -rf /tmp/gloo/*");
  }

  std::shared_ptr<cylon::CylonContext> ctx;
  if (!cylon::CylonContext::InitDistributed(config, &ctx).is_ok()) {
    return 1;
  }

  LOG(INFO) << "rank:" << ctx->GetRank() << " size:" << ctx->GetWorldSize();

  std::shared_ptr<cylon::Table> first_table, second_table, out;

  cylon::examples::create_two_in_memory_tables(kCount, kDup, ctx, first_table, second_table);

  cylon::join::config::JoinConfig jc{cylon::join::config::JoinType::INNER, 0, 0,
                                     cylon::join::config::JoinAlgorithm::SORT, "l_", "r_"};

  auto status = cylon::DistributedJoin(first_table, second_table, jc, out);

  if (!status.is_ok()) {
    LOG(INFO) << "Table join failed ";
    return 1;
  }

  LOG(INFO) << "First table had : " << first_table->Rows() << " and Second table had : "
            << second_table->Rows() << ", Joined has : " << out->Rows();
  return 0;
}


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

#ifndef CYLON_SRC_CYLON_OPS_KERNELS_JOIN_H_
#define CYLON_SRC_CYLON_OPS_KERNELS_JOIN_H_

#include <arrow/api.h>
#include <queue>

#include "ctx/cylon_context.hpp"

namespace cylon {
namespace kernel {
class JoinKernel {
 private:
  std::shared_ptr<CylonContext> ctx;
  std::shared_ptr<arrow::Schema> schema;
  const cylon::join::config::JoinConfig *join_config;
  std::queue<std::shared_ptr<arrow::Table>> left_tables{};
  std::queue<std::shared_ptr<arrow::Table>> right_tables{};

 public:
  JoinKernel(const std::shared_ptr<cylon::CylonContext> &ctx,
             const std::shared_ptr<arrow::Schema> &schema,
             const cylon::join::config::JoinConfig *join_config);
  void InsertTable(int tag, const std::shared_ptr<cylon::Table> &table);
  cylon::Status Finalize(std::shared_ptr<cylon::Table> &result);
};
}
}

#endif //CYLON_SRC_CYLON_OPS_KERNELS_JOIN_H_

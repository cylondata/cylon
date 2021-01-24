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

#ifndef CYLON_SRC_CYLON_OPS_MERGE_OP_H_
#define CYLON_SRC_CYLON_OPS_MERGE_OP_H_

#include <vector>
#include <map>

#include "parallel_op.hpp"
#include "partition_op.hpp"

namespace cylon {
class MergeOp : public Op {
 private:
  std::unordered_map<int32_t, std::vector<std::shared_ptr<arrow::Table>>> received_tables_;
 public:
  MergeOp(const std::shared_ptr<CylonContext> &ctx,
          const std::shared_ptr<arrow::Schema> &schema,
          int32_t id,
          const std::shared_ptr<ResultsCallback> &callback);

  bool Execute(int tag, std::shared_ptr<Table> &table) override;

  void OnParentsFinalized() override;

  bool Finalize() override;
};
}

#endif //CYLON_SRC_CYLON_OPS_MERGE_OP_H_

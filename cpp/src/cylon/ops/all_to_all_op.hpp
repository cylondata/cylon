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

#ifndef CYLON_SRC_CYLON_OPS_KERNELS_SHUFFLE_H_
#define CYLON_SRC_CYLON_OPS_KERNELS_SHUFFLE_H_

#include <arrow/arrow_all_to_all.hpp>
#include <chrono>

#include "parallel_op.hpp"

namespace cylon {

class AllToAllOpConfig {

};

class AllToAllOp : public Op {

 private:
  cylon::ArrowAllToAll *all_to_all_{};
  bool finish_called_;
  std::chrono::high_resolution_clock::time_point start;
  bool started_time = false;
  long exec_time = 0;
 public:
  AllToAllOp(const std::shared_ptr<CylonContext> &ctx,
             const std::shared_ptr<arrow::Schema> &schema,
             int id,
             const ResultsCallback &callback,
             const AllToAllOpConfig &config);

  ~AllToAllOp() override;

  bool IsComplete() override;

  bool Execute(int tag, std::shared_ptr<Table> &table) override;

  bool Finalize() override;

  void OnParentsFinalized() override;
};
}

#endif //CYLON_SRC_CYLON_OPS_KERNELS_SHUFFLE_H_

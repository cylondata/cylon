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

#include "parallel_op.hpp"
namespace cylon {

class ShuffleOpConfig {

};

class ShuffleOp : public Op {

 private:
  cylon::ArrowCallback *arrow_callback_;
  cylon::ArrowAllToAll *all_to_all_;
  bool communication_done = false;

 public:
  ShuffleOp(std::shared_ptr<cylon::CylonContext> ctx,
            std::shared_ptr<arrow::Schema> schema,
            int id,
            std::shared_ptr<ResultsCallback> callback,
            std::shared_ptr<ShuffleOpConfig> config);

  void Progress() override ;

  bool Execute(int tag, std::shared_ptr<Table> table) override;

  bool Finalize() override;

  void OnParentsFinalized() override ;
};
}

#endif //CYLON_SRC_CYLON_OPS_KERNELS_SHUFFLE_H_

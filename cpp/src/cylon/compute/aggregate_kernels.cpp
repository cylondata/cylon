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

#include <memory>
#include "cylon/compute/aggregate_kernels.hpp"

namespace cylon {
namespace compute {

std::shared_ptr<AggregationOp> MakeAggregationOpFromID(AggregationOpId id) {
  switch (id) {
    case SUM:return std::make_shared<SumOp>();
    case MIN:return std::make_shared<MinOp>();
    case MAX:return std::make_shared<MaxOp>();
    case COUNT:return std::make_shared<CountOp>();
    case MEAN:return std::make_shared<MeanOp>();
    case VAR:return std::make_shared<VarOp>();
    case STDDEV:return std::make_shared<StdDevOp>();
    case NUNIQUE:return std::make_shared<NUniqueOp>();
    case QUANTILE:return std::make_shared<QuantileOp>();
  }

  return nullptr;
}

}
}
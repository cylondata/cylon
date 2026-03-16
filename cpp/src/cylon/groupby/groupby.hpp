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

#ifndef CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_HPP_
#define CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_HPP_

#include <cylon/table.hpp>
#include <cylon/compute/aggregate_kernels.hpp>

namespace cylon {

static const std::vector<compute::AggregationOpId>
    ASSOCIATIVE_OPS{compute::SUM, compute::MIN, compute::MAX};

static inline bool is_associative(const std::vector<compute::AggregationOpId> &aggregate_ops) {
  return std::all_of(aggregate_ops.begin(), aggregate_ops.end(), [](const compute::AggregationOpId &op) {
    return std::find(ASSOCIATIVE_OPS.begin(), ASSOCIATIVE_OPS.end(), op) != ASSOCIATIVE_OPS.end();
  });
}

Status DistributedHashGroupBy(std::shared_ptr<Table> &table,
                              const std::vector<int32_t> &index_cols,
                              const std::vector<int32_t> &aggregate_cols,
                              const std::vector<compute::AggregationOpId> &aggregate_ops,
                              std::shared_ptr<Table> &output);

Status DistributedHashGroupBy(std::shared_ptr<Table> &table,
                              int32_t index_col,
                              const std::vector<int32_t> &aggregate_cols,
                              const std::vector<compute::AggregationOpId> &aggregate_ops,
                              std::shared_ptr<Table> &output);

Status DistributedPipelineGroupBy(std::shared_ptr<Table> &table,
                                  int32_t index_col,
                                  const std::vector<int32_t> &aggregate_cols,
                                  const std::vector<compute::AggregationOpId> &aggregate_ops,
                                  std::shared_ptr<Table> &output);

}

#endif //CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_HPP_

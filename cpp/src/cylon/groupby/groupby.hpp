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

#include <status.hpp>
#include <table.hpp>
#include <compute/aggregates.hpp>
#include <map>

#include "groupby_aggregate_ops.hpp"

namespace cylon {

Status GroupBy(std::shared_ptr<Table> &table,
               int64_t index_col,
               const std::vector<int64_t> &aggregate_cols,
               const std::vector<GroupByAggregationOp> &aggregate_ops,
               std::shared_ptr<Table> &output,
               bool use_local_combine = false);

Status PipelineGroupBy(std::shared_ptr<Table> &table,
               int64_t index_col,
               const std::vector<int64_t> &aggregate_cols,
               const std::vector<GroupByAggregationOp> &aggregate_ops,
               std::shared_ptr<Table> &output);

}

#endif //CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_HPP_

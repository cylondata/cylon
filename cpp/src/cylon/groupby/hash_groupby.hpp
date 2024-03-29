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

#ifndef CYLON_CPP_SRC_CYLON_GROUPBY_HASH_GROUPBY_HPP_
#define CYLON_CPP_SRC_CYLON_GROUPBY_HASH_GROUPBY_HPP_

#include <cylon/table.hpp>
#include <cylon/compute/aggregate_kernels.hpp>

namespace cylon {

/**
 * Hash group-by operation by using <col_index, AggregationOpId> pairs
 * NOTE: Nulls in the value columns will be ignored!
 * @param table
 * @param idx_cols
 * @param aggregate_cols
 * @param output
 * @return
 */
Status HashGroupBy(const std::shared_ptr<Table> &table,
                   const std::vector<int32_t> &idx_cols,
                   const std::vector<std::pair<int32_t, compute::AggregationOpId>> &aggregate_cols,
                   std::shared_ptr<Table> &output);

/**
 * Hash group-by operation by using <col_index, AggregationOp> pairs
 * NOTE: Nulls in the value columns will be ignored!
 * @param table
 * @param idx_cols
 * @param aggregations
 * @param output
 * @return
 */
Status HashGroupBy(const std::shared_ptr<Table> &table,
                   const std::vector<int32_t> &idx_cols,
                   const std::vector<std::pair<int32_t, std::shared_ptr<compute::AggregationOp>>> &aggregations,
                   std::shared_ptr<Table> &output);
/**
 * Hash group-by operation by using AggregationOpId vector
 * NOTE: Nulls in the value columns will be ignored!
 * @param table
 * @param idx_cols
 * @param aggregate_cols
 * @param aggregate_ops
 * @param output
 * @return
 */
Status HashGroupBy(std::shared_ptr<Table> &table,
                   const std::vector<int32_t> &idx_cols,
                   const std::vector<int32_t> &aggregate_cols,
                   const std::vector<compute::AggregationOpId> &aggregate_ops,
                   std::shared_ptr<Table> &output);
Status HashGroupBy(std::shared_ptr<Table> &table,
                   int32_t idx_col,
                   const std::vector<int32_t> &aggregate_cols,
                   const std::vector<compute::AggregationOpId> &aggregate_ops,
                   std::shared_ptr<Table> &output);

}

#endif //CYLON_CPP_SRC_CYLON_GROUPBY_HASH_GROUPBY_HPP_

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

#ifndef CYLON_CPP_SRC_CYLON_COMPUTE_SCALAR_AGGREGATE_HPP_
#define CYLON_CPP_SRC_CYLON_COMPUTE_SCALAR_AGGREGATE_HPP_

#include <arrow/api.h>

#include "cylon/status.hpp"
#include "cylon/compute/aggregate_kernels.hpp"
#include "cylon/net/comm_operations.hpp"
#include "cylon/ctx/cylon_context.hpp"
#include "cylon/scalar.hpp"
#include "cylon/column.hpp"

namespace cylon {
namespace compute {

/**
 * Reduce an array is a distributed fashion. It is done in the following stages.
 *  1. MapToGroups: Calculate group_ids for value_col
 *  2. CombineLocally: Combine value_col locally based on group_ids (which creates an intermediate array vector)
 *  3. Shuffle: Shuffle a temp table with intermediate results
 *  4. MapToGroups: Calculate group_ids for shuffled intermediate results
 *  5. ReduceShuffledResults: Reduce shuffled intermediate results (which creates a reduced array vector)
 *  6. Finalize: Finalize the reduced arrays
 *
 *  ex: take `mean` operation
 *  1. Calculate group_ids for value_col
 *  2. Locally calculate sum and count for each group_id (intermediate_arrays = {sums, cnts})
 *  3. Shuffle intermediate_array
 *  4. Calculate group_ids for shuffled intermediate results
 *  5. Reduce shuffled sums and counts individually (reduced_arrays = {reduced_sums, reduced_cnts})
 *  6. output = divide(reduced_sum/ reduced_cnts)
 *
 *  In a serial execution mode, this will be simplified into following stages.
 *  1. MapToGroups: Calculate group_ids for value_col
 *  2. CombineLocally: Combine value_col locally based on group_ids (which creates an intermediate array vector)
 *  3. Finalize: Finalize the intermediate arrays
 */
struct ScalarAggregateKernel {
 public:
  virtual ~ScalarAggregateKernel() = default;

  virtual void Init(arrow::MemoryPool *pool, compute::KernelOptions *options) = 0;

  /**
   * Combine `value_col` array locally based on the group_id, and push intermediate results to
   * `combined_results` array vector.
   * @param value_col
   * @param local_group_ids
   * @param local_num_groups
   * @param combined_results
   * @return
   */
  virtual Status CombineLocally(const std::shared_ptr<arrow::Array> &value_col,
                                std::shared_ptr<arrow::Array> *combined_results) const = 0;

  /**
   * Create the final output array
   * @param combined_results
   * @param output
   * @return
   */
  virtual Status Finalize(const std::shared_ptr<arrow::Array> &combined_results,
                          std::shared_ptr<arrow::Scalar> *output) const = 0;

  virtual net::ReduceOp reduce_op() const = 0;
};

Status ScalarAggregate(const std::shared_ptr<CylonContext> &ctx,
                       const std::unique_ptr<ScalarAggregateKernel> &kernel,
                       const std::shared_ptr<arrow::Array> &values,
                       std::shared_ptr<arrow::Scalar> *result,
                       compute::KernelOptions *kernel_options = NULLPTR);

Status Sum(const std::shared_ptr<CylonContext> &ctx,
           const std::shared_ptr<Column> &values,
           std::shared_ptr<Scalar> *result);

Status Min(const std::shared_ptr<CylonContext> &ctx,
           const std::shared_ptr<Column> &values,
           std::shared_ptr<Scalar> *result);

Status Max(const std::shared_ptr<CylonContext> &ctx,
           const std::shared_ptr<Column> &values,
           std::shared_ptr<Scalar> *result);

Status Count(const std::shared_ptr<CylonContext> &ctx,
             const std::shared_ptr<Column> &values,
             std::shared_ptr<Scalar> *result);

Status Mean(const std::shared_ptr<CylonContext> &ctx,
            const std::shared_ptr<Column> &values,
            std::shared_ptr<Scalar> *result);

}
}
#endif //CYLON_CPP_SRC_CYLON_COMPUTE_SCALAR_AGGREGATE_HPP_

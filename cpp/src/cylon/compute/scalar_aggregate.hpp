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
 *  1. CombineLocally: Combine values locally (which creates an intermediate array)
 *  2. AllReduce: All-reduce intermediate results
 *  3. Finalize: Finalize the intermediate results to produce a scalar
 */
struct ScalarAggregateKernel {
 public:
  virtual ~ScalarAggregateKernel() = default;

  virtual void Init(arrow::MemoryPool *pool, const KernelOptions *options) = 0;

  /**
   * Combine `values` array locally based on the group_id, and push intermediate results to
   * `combined_results` array vector.
   * @param values
   * @param local_group_ids
   * @param local_num_groups
   * @param combined_results
   * @return
   */
  virtual Status CombineLocally(const std::shared_ptr<arrow::Array> &values,
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
                       const KernelOptions *kernel_options = NULLPTR);

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

Status Variance(const std::shared_ptr<CylonContext> &ctx,
                const std::shared_ptr<Column> &values,
                std::shared_ptr<Scalar> *result,
                const VarKernelOptions &options = VarKernelOptions());

Status StdDev(const std::shared_ptr<CylonContext> &ctx,
              const std::shared_ptr<Column> &values,
              std::shared_ptr<Scalar> *result,
              const VarKernelOptions &options = VarKernelOptions());

}
}
#endif //CYLON_CPP_SRC_CYLON_COMPUTE_SCALAR_AGGREGATE_HPP_

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

#include <arrow/compute/api.h>

#include <utility>

#include "scalar_aggregate.hpp"
#include "cylon/ctx/arrow_memory_pool_utils.hpp"

namespace cylon {
namespace compute {

template<typename CombineFn, net::ReduceOp ReduceOp>
struct TrivialScalarAggregateKernelImpl : public ScalarAggregateKernel {

 public:
  explicit TrivialScalarAggregateKernelImpl(std::shared_ptr<arrow::DataType> in_type)
      : out_type(std::move(in_type)) {}

  size_t num_combined_results() const override { return 1; }
  net::ReduceOp reduce_op() const override { return ReduceOp; }
  const std::shared_ptr<arrow::DataType> &output_type() const override { return out_type; }

  void Init(arrow::MemoryPool *pool, compute::KernelOptions *options) override {
    CYLON_UNUSED(options);
    pool_ = pool;
  }

  Status CombineLocally(const std::shared_ptr<arrow::Array> &value_col,
                        std::shared_ptr<arrow::Array> *combined_results) const override {
    std::shared_ptr<arrow::Scalar> comb_scalar;
    RETURN_CYLON_STATUS_IF_FAILED(CombineFn::Combine(value_col, &comb_scalar));
    CYLON_ASSIGN_OR_RAISE(*combined_results, arrow::MakeArrayFromScalar(*comb_scalar, 1, pool_));
    return Status::OK();
  }

  Status Finalize(const std::shared_ptr<arrow::Array> &combined_results,
                  std::shared_ptr<arrow::Scalar> *output) const override {
    CYLON_ASSIGN_OR_RAISE(*output, combined_results->GetScalar(0));
    // sometimes the combined result may not be the same as out_type. Therefore, cast!
    CYLON_ASSIGN_OR_RAISE(*output, (*output)->CastTo(out_type));
    return Status::OK();
  }

 protected:
  const std::shared_ptr<arrow::DataType> out_type;
  arrow::MemoryPool *pool_ = nullptr;
};

struct SumFnWrapper {
  static Status Combine(const std::shared_ptr<arrow::Array> &value_col,
                        std::shared_ptr<arrow::Scalar> *comb_scalar) {
    CYLON_ASSIGN_OR_RAISE(auto res, arrow::compute::Sum(value_col))
    *comb_scalar = res.scalar();
    return Status::OK();
  }
};
struct SumKernelImpl : public TrivialScalarAggregateKernelImpl<SumFnWrapper, net::SUM> {
  std::string name() const override { return "sum"; }
};

template<bool take_min>
struct MinMaxFnWrapper {
  static Status Combine(const std::shared_ptr<arrow::Array> &value_col,
                        std::shared_ptr<arrow::Scalar> *comb_scalar) {
    CYLON_ASSIGN_OR_RAISE(auto res, arrow::compute::MinMax(value_col))
    const auto &min_max = res.scalar_as<arrow::StructScalar>().value;
    *comb_scalar = take_min ? min_max[0] : min_max[1];
    return Status::OK();
  }
};
struct MinKernelImpl : public TrivialScalarAggregateKernelImpl<MinMaxFnWrapper<true>, net::MIN> {
  std::string name() const override { return "min"; }
};
struct MaxKernelImpl : public TrivialScalarAggregateKernelImpl<MinMaxFnWrapper<false>, net::MAX> {
  std::string name() const override { return "max"; }
};

Status ScalarAggregate(const std::shared_ptr<CylonContext> &ctx,
                       const std::unique_ptr<ScalarAggregateKernel> &kernel,
                       const std::shared_ptr<arrow::Array> &values,
                       std::shared_ptr<arrow::Scalar> *result,
                       compute::KernelOptions *kernel_options) {
  auto pool = ToArrowPool(ctx);

  // init the kernel
  kernel->Init(pool, kernel_options);

  // locally combine
  std::shared_ptr<arrow::Array> combined_results;
  RETURN_CYLON_STATUS_IF_FAILED(kernel->CombineLocally(values, &combined_results));



  return Status();
}
}
}


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

#include <utility>

#include <arrow/compute/api.h>
#include <arrow/visitor_inline.h>

#include "cylon/compute/scalar_aggregate.hpp"
#include "cylon/ctx/arrow_memory_pool_utils.hpp"

namespace cylon {
namespace compute {

template<typename CombineFn, net::ReduceOp ReduceOp>
struct TrivialScalarAggregateKernelImpl : public ScalarAggregateKernel {

 public:
  explicit TrivialScalarAggregateKernelImpl(std::shared_ptr<arrow::DataType> out_type)
      : out_type_(std::move(out_type)) {}

  size_t num_combined_results() const override { return 1; }
  net::ReduceOp reduce_op() const override { return ReduceOp; }
  const std::shared_ptr<arrow::DataType> &output_type() const override { return out_type_; }

  void Init(arrow::MemoryPool *pool, compute::KernelOptions *options) override {
    CYLON_UNUSED(options);
    pool_ = pool;
  }

  Status CombineLocally(const std::shared_ptr<arrow::Array> &value_col,
                        std::shared_ptr<arrow::Array> *combined_results) const override {
    std::shared_ptr<arrow::Scalar> comb_scalar;
    RETURN_CYLON_STATUS_IF_FAILED(CombineFn::Combine(value_col, &comb_scalar));

    // arrow sum upcasts sum to Int64, UInt64, Float64. So, change back to original type
    CYLON_ASSIGN_OR_RAISE(comb_scalar, comb_scalar->CastTo(out_type_))
    CYLON_ASSIGN_OR_RAISE(*combined_results, arrow::MakeArrayFromScalar(*comb_scalar, 1, pool_))
    return Status::OK();
  }

  Status Finalize(const std::shared_ptr<arrow::Array> &combined_results,
                  std::shared_ptr<arrow::Scalar> *output) const override {
    CYLON_ASSIGN_OR_RAISE(*output, combined_results->GetScalar(0))
    return Status::OK();
  }

 protected:
  const std::shared_ptr<arrow::DataType> out_type_;
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
  explicit SumKernelImpl(std::shared_ptr<arrow::DataType> in_type)
      : TrivialScalarAggregateKernelImpl(std::move(in_type)) {}
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
  explicit MinKernelImpl(std::shared_ptr<arrow::DataType> in_type)
      : TrivialScalarAggregateKernelImpl(std::move(in_type)) {}
  std::string name() const override { return "min"; }
};
struct MaxKernelImpl : public TrivialScalarAggregateKernelImpl<MinMaxFnWrapper<false>, net::MAX> {
  explicit MaxKernelImpl(std::shared_ptr<arrow::DataType> in_type)
      : TrivialScalarAggregateKernelImpl(std::move(in_type)) {}
  std::string name() const override { return "max"; }
};

struct CountFnWrapper {
  static Status Combine(const std::shared_ptr<arrow::Array> &value_col,
                        std::shared_ptr<arrow::Scalar> *comb_scalar) {
    *comb_scalar = std::make_shared<arrow::Int64Scalar>(value_col->length());
    return Status::OK();
  }
};
struct CountKernelImpl : public TrivialScalarAggregateKernelImpl<CountFnWrapper, net::SUM> {
  explicit CountKernelImpl() : TrivialScalarAggregateKernelImpl(arrow::int64()) {}
  std::string name() const override { return "count"; }
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

  if (ctx->GetWorldSize() > 1) {
    const auto &comm = ctx->GetCommunicator();
    std::shared_ptr<Column> reduced;
    RETURN_CYLON_STATUS_IF_FAILED(comm->AllReduce(ctx, Column::Make(std::move(combined_results)),
                                                  kernel->reduce_op(), &reduced));

    RETURN_CYLON_STATUS_IF_FAILED(kernel->Finalize(reduced->data(), result));
  } else {
    RETURN_CYLON_STATUS_IF_FAILED(kernel->Finalize(combined_results, result));
  }
  return Status::OK();
}

Status run_scalar_aggregate(const std::shared_ptr<CylonContext> &ctx,
                            const std::unique_ptr<ScalarAggregateKernel> &kern,
                            const std::shared_ptr<Column> &values,
                            std::shared_ptr<Scalar> *result) {
  std::shared_ptr<arrow::Scalar> res;
  RETURN_CYLON_STATUS_IF_FAILED(ScalarAggregate(ctx, kern, values->data(), &res));
  *result = Scalar::Make(std::move(res));
  return Status::OK();
}

Status Sum(const std::shared_ptr<CylonContext> &ctx,
           const std::shared_ptr<Column> &values,
           std::shared_ptr<Scalar> *result) {
  const std::unique_ptr<ScalarAggregateKernel>
      &kern = std::make_unique<SumKernelImpl>(values->data()->type());
  return run_scalar_aggregate(ctx, kern, values, result);
}

Status Min(const std::shared_ptr<CylonContext> &ctx,
           const std::shared_ptr<Column> &values,
           std::shared_ptr<Scalar> *result) {
  const std::unique_ptr<ScalarAggregateKernel>
      &kern = std::make_unique<MinKernelImpl>(values->data()->type());
  return run_scalar_aggregate(ctx, kern, values, result);
}

Status Max(const std::shared_ptr<CylonContext> &ctx,
           const std::shared_ptr<Column> &values,
           std::shared_ptr<Scalar> *result) {
  const std::unique_ptr<ScalarAggregateKernel>
      &kern = std::make_unique<MaxKernelImpl>(values->data()->type());
  return run_scalar_aggregate(ctx, kern, values, result);
}

Status Count(const std::shared_ptr<CylonContext> &ctx,
             const std::shared_ptr<Column> &values,
             std::shared_ptr<Scalar> *result) {
  const std::unique_ptr<ScalarAggregateKernel> &kern = std::make_unique<CountKernelImpl>();
  return run_scalar_aggregate(ctx, kern, values, result);
}

}
}


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
#include <cmath>

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

  net::ReduceOp reduce_op() const override { return ReduceOp; }

  void Init(arrow::MemoryPool *pool, const KernelOptions *options) override {
    CYLON_UNUSED(options);
    pool_ = pool;
  }

  Status CombineLocally(const std::shared_ptr<arrow::Array> &values,
                        std::shared_ptr<arrow::Array> *combined_results) const override {
    std::shared_ptr<arrow::Scalar> comb_scalar;
    RETURN_CYLON_STATUS_IF_FAILED(CombineFn::Combine(values, &comb_scalar));

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
};
struct MaxKernelImpl : public TrivialScalarAggregateKernelImpl<MinMaxFnWrapper<false>, net::MAX> {
  explicit MaxKernelImpl(std::shared_ptr<arrow::DataType> in_type)
      : TrivialScalarAggregateKernelImpl(std::move(in_type)) {}
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
};

struct MeanKernelImpl : public ScalarAggregateKernel {
  using OutType = arrow::DoubleType;
  using c_type = OutType::c_type;

 public:
  void Init(arrow::MemoryPool *pool, const KernelOptions *options) override {
    CYLON_UNUSED(options);
    pool_ = pool;
  }

  // combined_results = [sum, count]
  Status CombineLocally(const std::shared_ptr<arrow::Array> &values,
                        std::shared_ptr<arrow::Array> *combined_results) const override {
    const auto &out_type_ = arrow::TypeTraits<OutType>::type_singleton();

    CYLON_ASSIGN_OR_RAISE(auto res, arrow::compute::Sum(values))
    auto sum_scalar = res.scalar();
    // cast to double
    CYLON_ASSIGN_OR_RAISE(sum_scalar, sum_scalar->CastTo(out_type_))

    // make an array of 2 with sum scalar
    CYLON_ASSIGN_OR_RAISE(*combined_results, arrow::MakeArrayFromScalar(*sum_scalar, 2, pool_))
    // set second value to count
    auto *mut_values = (*combined_results)->data()->GetMutableValues<c_type>(1);
    mut_values[1] = static_cast<c_type>(values->length());

    return Status::OK();
  }

  Status Finalize(const std::shared_ptr<arrow::Array> &combined_results,
                  std::shared_ptr<arrow::Scalar> *output) const override {
    assert(combined_results->type()->id() == OutType::type_id);
    assert(combined_results->length() == 2);
    const auto *values = combined_results->data()->GetValues<c_type>(1);
    *output = arrow::MakeScalar(values[0] / values[1]);
    return Status::OK();
  }

  net::ReduceOp reduce_op() const override {
    return net::SUM;
  }

 private:
  arrow::MemoryPool *pool_ = nullptr;
};

struct VarianceKernelImpl : public ScalarAggregateKernel {
 public:
  using OutType = arrow::DoubleType;
  using OutScalarType = arrow::DoubleScalar;
  using c_type = OutType::c_type;

  explicit VarianceKernelImpl(bool do_std = false) : do_std(do_std) {}

  void Init(arrow::MemoryPool *pool, const KernelOptions *options) override {
    pool_ = pool;
    ddof = reinterpret_cast<const VarKernelOptions *>(options)->ddof;
  }

  Status CombineLocally(const std::shared_ptr<arrow::Array> &values,
                        std::shared_ptr<arrow::Array> *combined_results) const override {
    const auto out_type_ = arrow::TypeTraits<OutType>::type_singleton();
    arrow::compute::ExecContext exec_ctx(pool_);

    // sum of squares
    CYLON_ASSIGN_OR_RAISE(auto x2_arr, arrow::compute::Power(values, arrow::MakeScalar(2),
                                                             arrow::compute::ArithmeticOptions(),
                                                             &exec_ctx))
    CYLON_ASSIGN_OR_RAISE(auto x2_sum_res, arrow::compute::Sum(x2_arr,
                                                               arrow::compute::ScalarAggregateOptions(),
                                                               &exec_ctx))
    CYLON_ASSIGN_OR_RAISE(auto x2_sum, x2_sum_res.scalar()->CastTo(out_type_))

    // sum
    CYLON_ASSIGN_OR_RAISE(auto sum_res, arrow::compute::Sum(values,
                                                            arrow::compute::ScalarAggregateOptions(),
                                                            &exec_ctx))
    CYLON_ASSIGN_OR_RAISE(auto sum, sum_res.scalar()->CastTo(out_type_))

    // create an array of 3 using sum of squares
    CYLON_ASSIGN_OR_RAISE(*combined_results, arrow::MakeArrayFromScalar(*x2_sum, 3, pool_))

    // set second value to sum and third value to count
    auto *mut_values = (*combined_results)->data()->GetMutableValues<c_type>(1);
    mut_values[1] = std::static_pointer_cast<OutScalarType>(sum)->value;
    mut_values[2] = static_cast<c_type>(values->length());

    return Status::OK();
  }

  Status Finalize(const std::shared_ptr<arrow::Array> &combined_results,
                  std::shared_ptr<arrow::Scalar> *output) const override {
    assert(combined_results->type()->id() == OutType::type_id);
    assert(combined_results->length() == 3);

    const auto *values = combined_results->data()->GetValues<c_type>(1);
    c_type x2_sum = values[0], sum = values[1], count = values[2], result;

    if (count == 1.) {
      result = 0;
    } else if (count != 0.) {
      c_type mean_sum_sq = x2_sum / count;
      c_type mean = sum / count;
      c_type var = (mean_sum_sq - mean * mean) * count / (count - c_type(ddof));
      result = do_std ? sqrt(var) : var;
    } else {
      return {Code::ValueError, "divide by 0 count"};
    }

    *output = arrow::MakeScalar(result);
    return Status::OK();
  }

  net::ReduceOp reduce_op() const override { return net::SUM; }

 private:
  int ddof = 0;
  bool do_std;
  arrow::MemoryPool *pool_ = nullptr;
};

Status ScalarAggregate(const std::shared_ptr<CylonContext> &ctx,
                       const std::unique_ptr<ScalarAggregateKernel> &kernel,
                       const std::shared_ptr<arrow::Array> &values,
                       std::shared_ptr<arrow::Scalar> *result,
                       const KernelOptions *kernel_options) {
  auto pool = ToArrowPool(ctx);

  // init the kernel
  kernel->Init(pool, kernel_options);

  // locally combine
  std::shared_ptr<arrow::Array> combined_results;
  RETURN_CYLON_STATUS_IF_FAILED(kernel->CombineLocally(values, &combined_results));

  if (ctx->GetWorldSize() > 1) {
    const auto &comm = ctx->GetCommunicator();
    std::shared_ptr<Column> reduced;
    RETURN_CYLON_STATUS_IF_FAILED(comm->AllReduce(Column::Make(std::move(combined_results)),
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
                            std::shared_ptr<Scalar> *result,
                            const compute::KernelOptions *kernel_options) {
  std::shared_ptr<arrow::Scalar> res;
  RETURN_CYLON_STATUS_IF_FAILED(ScalarAggregate(ctx, kern, values->data(), &res, kernel_options));
  *result = Scalar::Make(std::move(res));
  return Status::OK();
}

Status Sum(const std::shared_ptr<CylonContext> &ctx,
           const std::shared_ptr<Column> &values,
           std::shared_ptr<Scalar> *result) {
  const std::unique_ptr<ScalarAggregateKernel>
      &kern = std::make_unique<SumKernelImpl>(values->data()->type());
  return run_scalar_aggregate(ctx, kern, values, result, nullptr);
}

Status Min(const std::shared_ptr<CylonContext> &ctx,
           const std::shared_ptr<Column> &values,
           std::shared_ptr<Scalar> *result) {
  const std::unique_ptr<ScalarAggregateKernel>
      &kern = std::make_unique<MinKernelImpl>(values->data()->type());
  return run_scalar_aggregate(ctx, kern, values, result, nullptr);
}

Status Max(const std::shared_ptr<CylonContext> &ctx,
           const std::shared_ptr<Column> &values,
           std::shared_ptr<Scalar> *result) {
  const std::unique_ptr<ScalarAggregateKernel>
      &kern = std::make_unique<MaxKernelImpl>(values->data()->type());
  return run_scalar_aggregate(ctx, kern, values, result, nullptr);
}

Status Count(const std::shared_ptr<CylonContext> &ctx,
             const std::shared_ptr<Column> &values,
             std::shared_ptr<Scalar> *result) {
  const std::unique_ptr<ScalarAggregateKernel> &kern = std::make_unique<CountKernelImpl>();
  return run_scalar_aggregate(ctx, kern, values, result, nullptr);
}

Status Mean(const std::shared_ptr<CylonContext> &ctx,
            const std::shared_ptr<Column> &values,
            std::shared_ptr<Scalar> *result) {
  const std::unique_ptr<ScalarAggregateKernel> &kern = std::make_unique<MeanKernelImpl>();
  return run_scalar_aggregate(ctx, kern, values, result, nullptr);
}

Status Variance(const std::shared_ptr<CylonContext> &ctx,
                const std::shared_ptr<Column> &values,
                std::shared_ptr<Scalar> *result,
                const VarKernelOptions &options) {
  const std::unique_ptr<ScalarAggregateKernel> &kern = std::make_unique<VarianceKernelImpl>();
  return run_scalar_aggregate(ctx, kern, values, result, &options);
}

Status StdDev(const std::shared_ptr<CylonContext> &ctx,
              const std::shared_ptr<Column> &values,
              std::shared_ptr<Scalar> *result,
              const VarKernelOptions &options) {
  const std::unique_ptr<ScalarAggregateKernel>
      &kern = std::make_unique<VarianceKernelImpl>(/*do_std=*/true);
  return run_scalar_aggregate(ctx, kern, values, result, &options);
}

}
}


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

#include "cylon/compute/aggregates.hpp"
#include "cylon/ctx/arrow_memory_pool_utils.hpp"

namespace cylon {
namespace compute {

template<typename CombineFn, net::ReduceOp ReduceOp>
struct TrivialScalarAggregateKernelImpl : public ScalarAggregateKernel {

 public:
  TrivialScalarAggregateKernelImpl() = default;

  net::ReduceOp reduce_op() const override { return ReduceOp; }

  void Init(arrow::MemoryPool *pool, const KernelOptions *options) override {
    pool_ = pool;
    options_ = options;
  }

  Status CombineLocally(const std::shared_ptr<arrow::Array> &values,
                        std::shared_ptr<arrow::Array> *combined_results) const override {
    std::shared_ptr<arrow::Scalar> comb_scalar;
    RETURN_CYLON_STATUS_IF_FAILED(CombineFn::Combine(values, &comb_scalar, options_));

    const auto &out_type = GetOutputType(values->type());

    // arrow sum upcasts sum to Int64, UInt64, Float64. So, change back to original type
    CYLON_ASSIGN_OR_RAISE(comb_scalar, comb_scalar->CastTo(out_type))
    CYLON_ASSIGN_OR_RAISE(*combined_results, arrow::MakeArrayFromScalar(*comb_scalar, 1, pool_))
    return Status::OK();
  }

  Status Finalize(const std::shared_ptr<arrow::Array> &combined_results,
                  std::shared_ptr<arrow::Scalar> *output) const override {
    CYLON_ASSIGN_OR_RAISE(*output, combined_results->GetScalar(0))
    return Status::OK();
  }

  int32_t num_intermediate_results() const override { return 1; }

 protected:
  arrow::MemoryPool *pool_ = nullptr;
  const KernelOptions *options_ = nullptr;
};

struct SumFnWrapper {
  static Status Combine(const std::shared_ptr<arrow::Array> &value_col,
                        std::shared_ptr<arrow::Scalar> *comb_scalar,
                        const KernelOptions *options) {
    arrow::compute::ScalarAggregateOptions
        opt(reinterpret_cast<const BasicOptions *>(options)->skip_nulls_);
    CYLON_ASSIGN_OR_RAISE(auto res, arrow::compute::Sum(value_col, opt))
    *comb_scalar = res.scalar();
    return Status::OK();
  }
};
struct SumKernelImpl : public TrivialScalarAggregateKernelImpl<SumFnWrapper, net::SUM> {
  explicit SumKernelImpl() : TrivialScalarAggregateKernelImpl() {}
};

template<bool take_min>
struct MinMaxFnWrapper {
  static Status Combine(const std::shared_ptr<arrow::Array> &value_col,
                        std::shared_ptr<arrow::Scalar> *comb_scalar,
                        const KernelOptions *options) {
    arrow::compute::ScalarAggregateOptions
        opt(reinterpret_cast<const BasicOptions *>(options)->skip_nulls_);
    CYLON_ASSIGN_OR_RAISE(auto res, arrow::compute::MinMax(value_col, opt))
    const auto &min_max = res.scalar_as<arrow::StructScalar>().value;
    *comb_scalar = take_min ? min_max[0] : min_max[1];
    return Status::OK();
  }
};
struct MinKernelImpl : public TrivialScalarAggregateKernelImpl<MinMaxFnWrapper<true>, net::MIN> {
  explicit MinKernelImpl() : TrivialScalarAggregateKernelImpl() {}
};
struct MaxKernelImpl : public TrivialScalarAggregateKernelImpl<MinMaxFnWrapper<false>, net::MAX> {
  explicit MaxKernelImpl() : TrivialScalarAggregateKernelImpl() {}
};

struct CountFnWrapper {
  static Status Combine(const std::shared_ptr<arrow::Array> &value_col,
                        std::shared_ptr<arrow::Scalar> *comb_scalar,
                        const KernelOptions *options) {
    CYLON_UNUSED(options);
    *comb_scalar = std::make_shared<arrow::Int64Scalar>(value_col->length());
    return Status::OK();
  }
};
struct CountKernelImpl : public TrivialScalarAggregateKernelImpl<CountFnWrapper, net::SUM> {
  explicit CountKernelImpl() : TrivialScalarAggregateKernelImpl() {}

  std::shared_ptr<arrow::DataType>
  GetOutputType(const std::shared_ptr<arrow::DataType> &in_type) const override {
    CYLON_UNUSED(in_type);
    return arrow::int64();
  }
};

struct MeanKernelImpl : public ScalarAggregateKernel {
  using OutType = arrow::DoubleType;
  using c_type = OutType::c_type;

 public:
  void Init(arrow::MemoryPool *pool, const KernelOptions *options) override {
    skip_nulls_ = reinterpret_cast<const BasicOptions *>(options)->skip_nulls_;
    pool_ = pool;
  }

  // combined_results = [sum, count]
  Status CombineLocally(const std::shared_ptr<arrow::Array> &values,
                        std::shared_ptr<arrow::Array> *combined_results) const override {
    const auto &out_type_ = arrow::TypeTraits<OutType>::type_singleton();

    arrow::compute::ScalarAggregateOptions opt(skip_nulls_);
    CYLON_ASSIGN_OR_RAISE(auto res, arrow::compute::Sum(values, opt))
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

    // if there's a null, return null scalar
    if (combined_results->null_count()) {
      *output = arrow::MakeNullScalar(combined_results->type());
      return Status::OK();
    }

    const auto *values = combined_results->data()->GetValues<c_type>(1);
    *output = arrow::MakeScalar(values[0] / values[1]);
    return Status::OK();
  }

  net::ReduceOp reduce_op() const override { return net::SUM; }

  int32_t num_intermediate_results() const override { return 2; }

  std::shared_ptr<arrow::DataType>
  GetOutputType(const std::shared_ptr<arrow::DataType> &in_type) const override {
    CYLON_UNUSED(in_type);
    return arrow::TypeTraits<OutType>::type_singleton();
  }

 private:
  bool skip_nulls_ = true;
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
    auto var_opt = reinterpret_cast<const VarKernelOptions *>(options);
    skip_nulls_ = var_opt->skip_nulls_;
    ddof = var_opt->ddof_;
  }

  Status CombineLocally(const std::shared_ptr<arrow::Array> &values,
                        std::shared_ptr<arrow::Array> *combined_results) const override {
    const auto out_type_ = arrow::TypeTraits<OutType>::type_singleton();
    arrow::compute::ExecContext exec_ctx(pool_);
    arrow::compute::ScalarAggregateOptions opt(skip_nulls_);

    // sum of squares
    CYLON_ASSIGN_OR_RAISE(auto x2_arr, arrow::compute::Power(values, arrow::MakeScalar(2),
                                                             arrow::compute::ArithmeticOptions(),
                                                             &exec_ctx))
    CYLON_ASSIGN_OR_RAISE(auto x2_sum_res, arrow::compute::Sum(x2_arr, opt, &exec_ctx))
    CYLON_ASSIGN_OR_RAISE(auto x2_sum, x2_sum_res.scalar()->CastTo(out_type_))

    // sum
    CYLON_ASSIGN_OR_RAISE(auto sum_res, arrow::compute::Sum(values, opt, &exec_ctx))
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

    // if there's a null, return null scalar
    if (combined_results->null_count()) {
      *output = arrow::MakeNullScalar(combined_results->type());
      return Status::OK();
    }

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

  int32_t num_intermediate_results() const override { return 3; }

  std::shared_ptr<arrow::DataType>
  GetOutputType(const std::shared_ptr<arrow::DataType> &in_type) const override {
    CYLON_UNUSED(in_type);
    return arrow::TypeTraits<OutType>::type_singleton();
  }

 private:
  int ddof = 0;
  bool do_std;
  bool skip_nulls_ = true;
  arrow::MemoryPool *pool_ = nullptr;
};

Status is_all_valid(const std::shared_ptr<net::Communicator> &comm,
                    const std::shared_ptr<arrow::Array> &values,
                    bool *res) {
  const auto &null_count = Scalar::Make(std::make_shared<arrow::Int64Scalar>(values->null_count()));
  std::shared_ptr<Scalar> out;
  RETURN_CYLON_STATUS_IF_FAILED(comm->AllReduce(null_count, net::SUM, &out));
  *res = std::static_pointer_cast<arrow::Int64Scalar>(out->data())->value == 0;
  return Status::OK();
}

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

    bool all_valid;
    RETURN_CYLON_STATUS_IF_FAILED(is_all_valid(comm, combined_results, &all_valid));
    if (!all_valid) {
      // combined_results array has nulls. So, return Null scalar
      *result = arrow::MakeNullScalar(values->type());
      return Status::OK();
    }

    std::shared_ptr<Column> reduced;
    RETURN_CYLON_STATUS_IF_FAILED(comm->AllReduce(Column::Make(std::move(combined_results)),
                                                  kernel->reduce_op(), &reduced));

    RETURN_CYLON_STATUS_IF_FAILED(kernel->Finalize(reduced->data(), result));
  } else {
    RETURN_CYLON_STATUS_IF_FAILED(kernel->Finalize(combined_results, result));
  }
  return Status::OK();
}

// taken from
// https://github.com/apache/arrow/blob/c848f12122014aba9958a3910e2324661c3c2d7a/cpp/src/arrow/compute/kernels/codegen_internal.cc#L226
std::shared_ptr<arrow::DataType> PromoteDatatype(const std::vector<arrow::Type::type> &ids) {
  // if all types same and primitive, return that
  auto id0 = ids[0];
  for (size_t i = 1; i < ids.size(); i++) {
    auto id = ids[i];

    if (id != id0) goto non_uniform_types;

    if (!arrow::is_primitive(id)) {
      // can not handle non-primitive types
      return nullptr;
    }
  }
  // all types are same

  non_uniform_types:
  //
  for (const auto &id: ids) {
    if (!arrow::is_floating(id) && !arrow::is_integer(id)) {
      // a common numeric type is only possible if all types are numeric
      return nullptr;
    }
    if (id == arrow::Type::HALF_FLOAT) {
      // float16 arithmetic is not currently supported
      return nullptr;
    }
  }

  for (const auto &id: ids) {
    if (id == arrow::Type::DOUBLE) return arrow::float64();
  }

  for (const auto &id: ids) {
    if (id == arrow::Type::FLOAT) return arrow::float32();
  }

  int max_width_signed = 0, max_width_unsigned = 0;

  for (const auto &id: ids) {
    auto max_width = &(arrow::is_signed_integer(id) ? max_width_signed : max_width_unsigned);
    *max_width = std::max(arrow::bit_width(id), *max_width);
  }

  if (max_width_signed == 0) {
    if (max_width_unsigned >= 64) return arrow::uint64();
    if (max_width_unsigned == 32) return arrow::uint32();
    if (max_width_unsigned == 16) return arrow::uint16();
    assert(max_width_unsigned == 8);
    return arrow::uint8();
  }

  if (max_width_signed <= max_width_unsigned) {
    max_width_signed = static_cast<int>(arrow::BitUtil::NextPower2(max_width_unsigned + 1));
  }

  if (max_width_signed >= 64) return arrow::int64();
  if (max_width_signed == 32) return arrow::int32();
  if (max_width_signed == 16) return arrow::int16();
  assert(max_width_signed == 8);
  return arrow::int8();
}

Status ScalarAggregate(const std::shared_ptr<CylonContext> &ctx,
                       const std::unique_ptr<ScalarAggregateKernel> &kernel,
                       const std::shared_ptr<arrow::Table> &table,
                       std::shared_ptr<arrow::Array> *result,
                       const KernelOptions *kernel_options) {
  auto pool = ToArrowPool(ctx);

  const int num_results = table->num_columns();

  auto table_ = table;
  if (util::CheckArrowTableContainsChunks(table)) {
    CYLON_ASSIGN_OR_RAISE(table_, table->CombineChunks(pool))
  }

  // promote the data type
  std::vector<arrow::Type::type> ids;
  ids.reserve(num_results);
  for (const auto &f: table->schema()->fields()) {
    ids.push_back(kernel->GetOutputType(f->type())->id());
  }
  const auto &promoted_type = PromoteDatatype(ids);
  if (promoted_type == nullptr) {
    return {Code::Invalid, "Unable to promote datatypes"};
  }

  // init the kernel
  kernel->Init(pool, kernel_options);

  // locally combine each array and create a single array
  arrow::ArrayVector combined_arrays;
  combined_arrays.reserve(num_results);
  for (const auto &values: table->columns()) {
    std::shared_ptr<arrow::Array> comb;
    RETURN_CYLON_STATUS_IF_FAILED(kernel->CombineLocally(values->chunk(0), &comb));

    if (!comb->type()->Equals(promoted_type)) {
      CYLON_ASSIGN_OR_RAISE(auto res, arrow::compute::Cast(comb, promoted_type))
      combined_arrays.push_back(res.make_array());
    } else {
      combined_arrays.push_back(std::move(comb));
    }
  }
  CYLON_ASSIGN_OR_RAISE(auto combined_results, arrow::Concatenate(combined_arrays, pool))
  assert(num_results * kernel->num_intermediate_results() == combined_results->length());

  if (ctx->GetWorldSize() > 1) {
    // all reduce combined_results
    const auto &comm = ctx->GetCommunicator();

    bool all_valid;
    RETURN_CYLON_STATUS_IF_FAILED(is_all_valid(comm, combined_results, &all_valid));
    if (!all_valid) {
      // combined_results array has nulls. So, return Null scalar
      CYLON_ASSIGN_OR_RAISE(*result,
                            arrow::MakeArrayOfNull(promoted_type, table_->num_columns(), pool))
      return Status::OK();
    }

    std::shared_ptr<Column> reduced;
    RETURN_CYLON_STATUS_IF_FAILED(comm->AllReduce(Column::Make(std::move(combined_results)),
                                                  kernel->reduce_op(), &reduced));
    combined_results = reduced->data();
  }

  std::unique_ptr<arrow::ArrayBuilder> builder;
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(arrow::MakeBuilder(pool, promoted_type, &builder));
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder->Reserve(num_results));
  const int results_per_col = kernel->num_intermediate_results();
  for (int i = 0; i < num_results; i++) {
    std::shared_ptr<arrow::Scalar> scalar;
    RETURN_CYLON_STATUS_IF_FAILED(kernel->Finalize(combined_results->Slice(i * results_per_col,
                                                                           results_per_col),
                                                   &scalar));
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder->AppendScalar(*scalar));
  }

  CYLON_ASSIGN_OR_RAISE(*result, builder->Finish());
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

Status run_scalar_aggregate(const std::shared_ptr<CylonContext> &ctx,
                            const std::unique_ptr<ScalarAggregateKernel> &kern,
                            const std::shared_ptr<Table> &values,
                            std::shared_ptr<Column> *result,
                            const compute::KernelOptions *kernel_options) {
  std::shared_ptr<arrow::Array> res;
  RETURN_CYLON_STATUS_IF_FAILED(ScalarAggregate(ctx,
                                                kern,
                                                values->get_table(),
                                                &res,
                                                kernel_options));
  *result = Column::Make(std::move(res));
  return Status::OK();
}

Status Sum(const std::shared_ptr<CylonContext> &ctx,
           const std::shared_ptr<Column> &values,
           std::shared_ptr<Scalar> *result,
           const BasicOptions &options) {
  const std::unique_ptr<ScalarAggregateKernel> &kern = std::make_unique<SumKernelImpl>();
  return run_scalar_aggregate(ctx, kern, values, result, &options);
}
Status Sum(const std::shared_ptr<CylonContext> &ctx,
           const std::shared_ptr<Table> &table,
           std::shared_ptr<Column> *result,
           const BasicOptions &options) {
  const std::unique_ptr<ScalarAggregateKernel> &kern = std::make_unique<SumKernelImpl>();
  return run_scalar_aggregate(ctx, kern, table, result, &options);
}

Status Min(const std::shared_ptr<CylonContext> &ctx,
           const std::shared_ptr<Column> &values,
           std::shared_ptr<Scalar> *result,
           const BasicOptions &options) {
  const std::unique_ptr<ScalarAggregateKernel> &kern = std::make_unique<MinKernelImpl>();
  return run_scalar_aggregate(ctx, kern, values, result, &options);
}
Status Min(const std::shared_ptr<CylonContext> &ctx,
           const std::shared_ptr<Table> &table,
           std::shared_ptr<Column> *result,
           const BasicOptions &options) {
  const std::unique_ptr<ScalarAggregateKernel> &kern = std::make_unique<MinKernelImpl>();
  return run_scalar_aggregate(ctx, kern, table, result, &options);
}

Status Max(const std::shared_ptr<CylonContext> &ctx,
           const std::shared_ptr<Column> &values,
           std::shared_ptr<Scalar> *result,
           const BasicOptions &options) {
  const std::unique_ptr<ScalarAggregateKernel> &kern = std::make_unique<MaxKernelImpl>();
  return run_scalar_aggregate(ctx, kern, values, result, &options);
}
Status Max(const std::shared_ptr<CylonContext> &ctx,
           const std::shared_ptr<Table> &table,
           std::shared_ptr<Column> *result,
           const BasicOptions &options) {
  const std::unique_ptr<ScalarAggregateKernel> &kern = std::make_unique<MaxKernelImpl>();
  return run_scalar_aggregate(ctx, kern, table, result, &options);
}

Status Count(const std::shared_ptr<CylonContext> &ctx,
             const std::shared_ptr<Column> &values,
             std::shared_ptr<Scalar> *result) {
  const std::unique_ptr<ScalarAggregateKernel> &kern = std::make_unique<CountKernelImpl>();
  return run_scalar_aggregate(ctx, kern, values, result, nullptr);
}
Status Count(const std::shared_ptr<CylonContext> &ctx,
             const std::shared_ptr<Table> &table,
             std::shared_ptr<Column> *result) {
  const std::unique_ptr<ScalarAggregateKernel> &kern = std::make_unique<CountKernelImpl>();
  return run_scalar_aggregate(ctx, kern, table, result, nullptr);
}

Status Mean(const std::shared_ptr<CylonContext> &ctx,
            const std::shared_ptr<Column> &values,
            std::shared_ptr<Scalar> *result,
            const BasicOptions &options) {
  const std::unique_ptr<ScalarAggregateKernel> &kern = std::make_unique<MeanKernelImpl>();
  return run_scalar_aggregate(ctx, kern, values, result, &options);
}
Status Mean(const std::shared_ptr<CylonContext> &ctx,
            const std::shared_ptr<Table> &table,
            std::shared_ptr<Column> *result,
            const BasicOptions &options) {
  const std::unique_ptr<ScalarAggregateKernel> &kern = std::make_unique<MeanKernelImpl>();
  return run_scalar_aggregate(ctx, kern, table, result, &options);
}

Status Variance(const std::shared_ptr<CylonContext> &ctx,
                const std::shared_ptr<Column> &values,
                std::shared_ptr<Scalar> *result,
                const VarKernelOptions &options) {
  const std::unique_ptr<ScalarAggregateKernel> &kern = std::make_unique<VarianceKernelImpl>();
  return run_scalar_aggregate(ctx, kern, values, result, &options);
}
Status Variance(const std::shared_ptr<CylonContext> &ctx,
                const std::shared_ptr<Table> &table,
                std::shared_ptr<Column> *result,
                const VarKernelOptions &options) {
  const std::unique_ptr<ScalarAggregateKernel> &kern = std::make_unique<VarianceKernelImpl>();
  return run_scalar_aggregate(ctx, kern, table, result, &options);
}

Status StdDev(const std::shared_ptr<CylonContext> &ctx,
              const std::shared_ptr<Column> &values,
              std::shared_ptr<Scalar> *result,
              const VarKernelOptions &options) {
  const std::unique_ptr<ScalarAggregateKernel>
      &kern = std::make_unique<VarianceKernelImpl>(/*do_std=*/true);
  return run_scalar_aggregate(ctx, kern, values, result, &options);
}
Status StdDev(const std::shared_ptr<CylonContext> &ctx,
              const std::shared_ptr<Table> &table,
              std::shared_ptr<Column> *result,
              const VarKernelOptions &options) {
  const std::unique_ptr<ScalarAggregateKernel>
      &kern = std::make_unique<VarianceKernelImpl>(/*do_std=*/true);
  return run_scalar_aggregate(ctx, kern, table, result, &options);
}

}
}


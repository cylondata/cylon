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

#include <arrow/api.h>
#include <status.hpp>
#include <table.hpp>

#include "../compute/compute_kernels.hpp"

namespace cylon {

template<compute::AggregationOp aggOp, typename ARROW_T, typename = typename std::enable_if<
    arrow::is_number_type<ARROW_T>::value | arrow::is_boolean_type<ARROW_T>::value>::type>
static Status aggregate(arrow::MemoryPool *pool,
                        const std::shared_ptr<arrow::Array> &arr,
                        const std::shared_ptr<arrow::Field> &field,
                        const std::vector<int64_t> &group_ids,
                        int64_t unique_groups,
                        std::shared_ptr<arrow::Array> &agg_array,
                        std::shared_ptr<arrow::Field> &agg_field) {
  if (arr->length() != (int64_t) group_ids.size()) {
    return Status(Code::Invalid, "group IDs != array length");
  }

  using C_TYPE = typename ARROW_T::c_type;
  using ARRAY_T = typename arrow::TypeTraits<ARROW_T>::ArrayType;

  using State = typename compute::KernelTraits<aggOp, C_TYPE>::State;
  using ResultT = typename compute::KernelTraits<aggOp, C_TYPE>::ResultT;
  const std::unique_ptr<compute::Kernel> &op = compute::CreateAggregateKernel<aggOp, C_TYPE>();

  State state;
  op->Init(&state); // initialize statte

  std::vector<State> agg_results(unique_groups, state); // initialize aggregates
  const std::shared_ptr<ARRAY_T> &carr = std::static_pointer_cast<ARRAY_T>(arr);
  for (int64_t i = 0; i < arr->length(); i++) {
    auto val = carr->Value(i);
    op->Update(&val, &agg_results[group_ids[i]]);
  }

  // need to create a builder from the ResultT, which is a C type
  using BUILDER_T = typename arrow::TypeTraits<typename arrow::CTypeTraits<ResultT>::ArrowType>::BuilderType;
  BUILDER_T builder(pool);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.Reserve(unique_groups))
  for (int64_t i = 0; i < unique_groups; i++) {
    ResultT res;
    op->Finalize(&agg_results[i], &res);
    builder.UnsafeAppend(res);
  }

  RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.Finish(&agg_array))

  const char *prefix = compute::KernelTraits<aggOp, C_TYPE>::name();
  agg_field = field->WithName(std::string(prefix) + field->name());

  return Status::OK();
}

using AggregationOp = std::function<Status(arrow::MemoryPool *pool,
                                           const std::shared_ptr<arrow::Array> &arr,
                                           const std::shared_ptr<arrow::Field> &field,
                                           const std::vector<int64_t> &group_ids,
                                           int64_t unique_groups,
                                           std::shared_ptr<arrow::Array> &agg_array,
                                           std::shared_ptr<arrow::Field> &agg_field)>;

template<typename ARROW_T, typename = typename std::enable_if<
    arrow::is_number_type<ARROW_T>::value | arrow::is_boolean_type<ARROW_T>::value>::type>
static inline AggregationOp resolve_op(const compute::AggregationOp &aggOp) {
  switch (aggOp) {
    case compute::SUM: return &aggregate<compute::SUM, ARROW_T>;
    case compute::COUNT: return &aggregate<compute::COUNT, ARROW_T>;
    case compute::MIN:return &aggregate<compute::MIN, ARROW_T>;
    case compute::MAX: return &aggregate<compute::MAX, ARROW_T>;
    case compute::MEAN: return &aggregate<compute::MEAN, ARROW_T>;
    default: return nullptr;
  }
}

Status HashGroupBy(const std::shared_ptr<Table> &table, const std::vector<int32_t> &idx_cols,
                   const std::vector<std::pair<int64_t, compute::AggregationOp>> &aggregate_cols,
                   std::shared_ptr<Table> &output);

}

#endif //CYLON_CPP_SRC_CYLON_GROUPBY_HASH_GROUPBY_HPP_

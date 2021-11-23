//
// Created by niranda on 11/22/21.
//

#include <cylon/thridparty/flat_hash_map/bytell_hash_map.hpp>
#include "cylon/mapreduce/mapreduce.hpp"
#include "cylon/util/macros.hpp"

#include <arrow/buffer_builder.h>
#include <arrow/visitor_inline.h>

namespace cylon {
namespace mapred {

template<typename ArrowT, typename CombineVisitor>
Status CombineVisitor(const std::shared_ptr<arrow::Array> &value_col,
                      const int64_t *local_group_ids,
                      CombineVisitor &&visitor) {
  using T = typename ArrowT::c_type;

  int64_t i = 0;
  arrow::VisitArrayDataInline<ArrowT>(*value_col->data(),
                                      [&](const T &val) {
                                        int64_t gid = local_group_ids[i];
                                        visitor(val, gid);
                                        i++;
                                      },
                                      [&]() {
                                        i++;
                                      });
  return Status::OK();
}

template<typename ArrowT, typename ReduceVisitor>
Status ReduceVisitor(const std::shared_ptr<CylonContext> &ctx, int64_t groups,
                     std::shared_ptr<arrow::Array> *output, ReduceVisitor &&visitor) {
  using BuilderT = typename arrow::TypeTraits<ArrowT>::BuilderType;

  BuilderT out_build(ToArrowPool(ctx));
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(out_build.Reserve(groups));

  for (int64_t i = 0; i < groups; i++) {
    out_build.UnsafeAppend(visitor(i));
  }

  CYLON_ASSIGN_OR_RAISE(*output, out_build.Finalize())
  return Status::OK();
}

Status MakeStructArray() {

}

template<typename ArrowT>
struct MeanKernelImpl : public MapReduceKernel {
  using T = typename ArrowT::c_type;
  using ResultT = T;
  using BuilderT = typename arrow::TypeTraits<ArrowT>::BuilderType;

  Status Init(const std::shared_ptr<CylonContext> &ctx, int64_t local_num_groups,
              arrow::ArrayVector *combined_results) const override {
    auto pool = ToArrowPool(ctx);

    combined_results->reserve(2);

    arrow::Int64Builder count_build(pool);
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(count_build.AppendEmptyValues(local_num_groups));
    CYLON_ASSIGN_OR_RAISE(auto counts, count_build.Finish())
    combined_results->push_back(std::move(counts));

    BuilderT sum_build(pool);
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(sum_build.AppendEmptyValues(local_num_groups));
    CYLON_ASSIGN_OR_RAISE(auto sums, sum_build.Finish())
    combined_results->push_back(std::move(sums));

    return Status::OK();
  }

  Status Combine(const std::shared_ptr<CylonContext> &ctx,
                 const std::shared_ptr<arrow::Array> &value_col,
                 const int64_t *local_group_ids,
                 int64_t local_num_groups,
                 arrow::ArrayVector *combined_results) const override {
    CYLON_UNUSED(ctx);
    CYLON_UNUSED(local_num_groups);

    T *sums = (*combined_results)[0]->data()->template GetMutableValues<T>(1);
    auto *counts = (*combined_results)[1]->data()->template GetMutableValues<int64_t>(1);

    return CombineVisitor<ArrowT>(value_col, local_group_ids,
                                  [&](const T &val, int64_t gid) {
                                    sums[gid] += val;
                                    counts[gid]++;
                                  });
  }

  Status Finalize(const std::shared_ptr<CylonContext> &ctx,
                  const arrow::ArrayVector &combined_results,
                  std::shared_ptr<arrow::Array> *output) const override {
    auto *sums = combined_results[0]->data()->template GetValues<T>(1);
    auto *counts = combined_results[1]->data()->template GetValues<int64_t>(1);
    int64_t groups = combined_results[0]->length();

    return ReduceVisitor<ArrowT>(ctx, groups, output,
                                 [&](int64_t i) {
                                   return sums[i] / counts[i];
                                 });
  };
};

template<typename ArrowT, typename Op>
struct MapReduceKernelImpl : public MapReduceKernel {
  using T = typename ArrowT::c_type;
  using ResultT = ArrowT;
  using BuilderT = typename arrow::TypeTraits<ArrowT>::BuilderType;

  Status Init(const std::shared_ptr<CylonContext> &ctx, int64_t local_num_groups,
              arrow::ArrayVector *combined_results) const override {
    return Op::Init(ctx, local_num_groups, combined_results);
  }

  Status Combine(const std::shared_ptr<CylonContext> &ctx,
                 const std::shared_ptr<arrow::Array> &value_col,
                 const int64_t *local_group_ids,
                 int64_t local_num_groups,
                 arrow::ArrayVector *combined_results) const override {
    CYLON_UNUSED(ctx);
    CYLON_UNUSED(local_num_groups);

    T *sums = (*combined_results)[0]->data()->template GetMutableValues<T>(1);
    auto *counts = (*combined_results)[1]->data()->template GetMutableValues<int64_t>(1);

    return CombineVisitor<ArrowT>(value_col, local_group_ids,
                                  [&](const T &val, int64_t gid) {
                                    sums[gid] += val;
                                    counts[gid]++;
                                  });
  }

  Status Reduce(const std::shared_ptr<CylonContext> &ctx,
                const arrow::ArrayVector &combined_results,
                const int64_t *local_group_ids,
                int64_t local_num_groups,
                arrow::ArrayVector *reduced_results) const override {
    return Status();
  }

  Status Finalize(const std::shared_ptr<CylonContext> &ctx,
                  const arrow::ArrayVector &reduced_results,
                  std::shared_ptr<arrow::Array> *output) const override {
    auto sums = reduced_results[0]->data()->template GetValues<T>(1);
    auto *counts = reduced_results[1]->data()->template GetValues<int64_t>(1);
    int64_t groups = reduced_results[0]->length();

    return ReduceVisitor<ResultT>(ctx, groups, output,
                                  [&](int64_t i) {
                                    return sums[i] / counts[i];
                                  });
  };
};

Status MapToGroupKernel::Map(const std::shared_ptr<CylonContext> &ctx,
                             const arrow::ArrayVector &arrays,
                             std::shared_ptr<arrow::Array> *local_group_ids,
                             std::shared_ptr<arrow::Array> *local_group_indices,
                             int64_t *local_num_groups) const {
  const int64_t num_rows = arrays[0]->length();
  if (std::any_of(arrays.begin() + 1, arrays.end(),
                  [&](const auto &arr) { return arr->length() != num_rows; })) {
    return {Code::Invalid, "array lengths should be the same"};
  }
  
  auto pool = ToArrowPool(ctx);

  // if empty, return
  if (num_rows == 0) {
    CYLON_ASSIGN_OR_RAISE(*local_group_ids, arrow::MakeArrayOfNull(arrow::int64(), 0, pool))
    *local_group_indices = *local_group_ids; // copy empty array
    *local_num_groups = 0;
    return Status::OK();
  }

  std::unique_ptr<TableRowIndexEqualTo> comp;
  RETURN_CYLON_STATUS_IF_FAILED(TableRowIndexEqualTo::Make(arrays, &comp));

  std::unique_ptr<TableRowIndexHash> hash;
  RETURN_CYLON_STATUS_IF_FAILED(TableRowIndexHash::Make(arrays, &hash));

  ska::bytell_hash_map<int64_t, int64_t, TableRowIndexHash, TableRowIndexEqualTo>
      hash_map(num_rows, *hash, *comp);

  arrow::Int64Builder group_ids_build(pool);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(group_ids_build.Reserve(num_rows));

  arrow::Int64Builder filter_build(pool);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED((filter_build.Reserve(num_rows)));

  int64_t unique = 0;
  for (int64_t i = 0; i < num_rows; i++) {
    const auto &res = hash_map.emplace(i, unique);
    if (res.second) { // this was a unique group
      group_ids_build.UnsafeAppend(unique);
      unique++;
      filter_build.UnsafeAppend(i);
    } else {
      group_ids_build.UnsafeAppend(res.first->second);
    }
  }

  RETURN_CYLON_STATUS_IF_ARROW_FAILED((group_ids_build.Finish(local_group_ids)));
  RETURN_CYLON_STATUS_IF_ARROW_FAILED((filter_build.Finish(local_group_indices)));
  *local_num_groups = unique;
  return Status::OK();
}
}
}
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

template<typename ArrowT, typename Visitor>
Status CombineVisit(const std::shared_ptr<arrow::Array> &value_col, const int64_t *local_group_ids,
                    Visitor &&visitor) {
  using T = typename ArrowT::c_type;
  int64_t i = 0;
  arrow::VisitArrayDataInline<ArrowT>(*value_col->data(),
                                      [&](const T &val) {
                                        int64_t gid = local_group_ids[i];
                                        visitor(val, gid);
                                        i++;
                                      },
                                      [&]() { i++; });
  return Status::OK();
}

template<typename ArrowT, typename Visitor>
Status FinalizeVisit(arrow::MemoryPool *pool, int64_t groups,
                     std::shared_ptr<arrow::Array> *output, Visitor &&visitor) {
  using BuilderT = typename arrow::TypeTraits<ArrowT>::BuilderType;

  BuilderT out_build(pool);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(out_build.Reserve(groups));
  for (int64_t i = 0; i < groups; i++) {
    out_build.UnsafeAppend(visitor(i));
  }
  CYLON_ASSIGN_OR_RAISE(*output, out_build.Finish())
  return Status::OK();
}

template<typename ArrowT, template<typename> class CombineOp>
struct MapReduceKernelImpl1D : public MapReduceKernel {
  using T = typename ArrowT::c_type;
  using BuilderT = typename arrow::TypeTraits<ArrowT>::BuilderType;

  size_t num_arrays() const override { return 1; }

  Status Init(const std::shared_ptr<CylonContext> &ctx, int64_t local_num_groups,
              arrow::ArrayVector *combined_results) override {
    auto pool = ToArrowPool(ctx);

    BuilderT res_build(pool);
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(res_build.AppendEmptyValues(local_num_groups));
    CYLON_ASSIGN_OR_RAISE(auto res, res_build.Finish())

    combined_results->push_back(std::move(res));
    return Status::OK();
  }

  Status Combine(const std::shared_ptr<arrow::Array> &value_col, const int64_t *local_group_ids,
                 int64_t local_num_groups, arrow::ArrayVector *combined_results) const override {
    assert(combined_results->size() == 1);
    assert(local_num_groups == (*combined_results)[0]->length());

    T *res = (*combined_results)[0]->data()->template GetMutableValues<T>(1);
    return CombineVisit<ArrowT>(value_col, local_group_ids,
                                [&](const T &val, int64_t gid) {
                                  res[gid] = CombineOp<T>::Call(res[gid], val);
                                });
  }

  Status Reduce(const arrow::ArrayVector &combined_results,
                const int64_t *local_group_ids,
                const int64_t *local_group_indices,
                int64_t local_num_groups,
                arrow::ArrayVector *reduced_results) const override {
    CYLON_UNUSED(local_group_indices);
    return Combine(combined_results[0], local_group_ids, local_num_groups, reduced_results);
  }

  Status Finalize(const arrow::ArrayVector &combined_results,
                  std::shared_ptr<arrow::Array> *output) const override {
    *output = combined_results[0];
    return Status::OK();
  };
};

template<typename T>
struct SumFunction {
  static T Call(const T &x, const T &y) { return x + y; };
};
template<typename ArrowT>
struct SumKernelImpl : public MapReduceKernelImpl1D<ArrowT, SumFunction> {};

template<typename T>
struct MinFunction {
  static T Call(const T &x, const T &y) { return std::min(x, y); };
};
template<typename ArrowT>
struct MinKernelImpl : public MapReduceKernelImpl1D<ArrowT, MinFunction> {};

template<typename T>
struct MaxFunction {
  static T Call(const T &x, const T &y) { return std::min(x, y); };
};
template<typename ArrowT>
struct MaxKernelImpl : public MapReduceKernelImpl1D<ArrowT, MaxFunction> {};

struct CountKernelImpl : public MapReduceKernel {
  size_t num_arrays() const override { return 1; }

  Status Init(const std::shared_ptr<CylonContext> &ctx,
              int64_t local_num_groups,
              arrow::ArrayVector *combined_results) override {
    arrow::Int64Builder count_build(ToArrowPool(ctx));
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(count_build.AppendEmptyValues(local_num_groups));
    CYLON_ASSIGN_OR_RAISE(auto counts, count_build.Finish());

    combined_results->push_back(std::move(counts));
    return Status::OK();
  }
  Status Combine(const std::shared_ptr<arrow::Array> &value_col,
                 const int64_t *local_group_ids,
                 int64_t local_num_groups,
                 arrow::ArrayVector *combined_results) const override {
    assert(local_num_groups == (*combined_results)[0]->length());
    auto *counts = (*combined_results)[0]->data()->template GetMutableValues<int64_t>(1);
    for (int64_t i = 0; i < value_col->length(); i++) {
      counts[local_group_ids[i]]++;
    }
    return Status::OK();
  }
  Status Reduce(const arrow::ArrayVector &combined_results,
                const int64_t *local_group_ids,
                const int64_t *local_group_indices,
                int64_t local_num_groups,
                arrow::ArrayVector *reduced_results) const override {
    CYLON_UNUSED(local_group_indices);
    assert(local_num_groups == (*reduced_results)[0]->length());
    auto *res = (*reduced_results)[0]->data()->template GetMutableValues<int64_t>(1);
    return CombineVisit<arrow::Int64Type>(combined_results[0], local_group_ids,
                                          [&](const int64_t &val, int64_t gid) {
                                            res[gid] += val;
                                          });
  }
  Status Finalize(const arrow::ArrayVector &reduced_results,
                  std::shared_ptr<arrow::Array> *output) const override {
    *output = reduced_results[0];
    return Status::OK();
  }
};

template<typename ArrowT>
struct MeanKernelImpl : public MapReduceKernel {
  using T = typename ArrowT::c_type;
  using ResultT = ArrowT;
  using BuilderT = typename arrow::TypeTraits<ArrowT>::BuilderType;

  arrow::MemoryPool *pool = nullptr;
  size_t num_arrays() const override { return 2; }

  Status Init(const std::shared_ptr<CylonContext> &ctx, int64_t local_num_groups,
              arrow::ArrayVector *combined_results) override {
    pool = ToArrowPool(ctx);

    arrow::Int64Builder count_build(pool);
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(count_build.AppendEmptyValues(local_num_groups));
    CYLON_ASSIGN_OR_RAISE(auto counts, count_build.Finish())

    BuilderT sum_build(pool);
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(sum_build.AppendEmptyValues(local_num_groups));
    CYLON_ASSIGN_OR_RAISE(auto sums, sum_build.Finish())

    combined_results->reserve(num_arrays());
    combined_results->push_back(std::move(sums));
    combined_results->push_back(std::move(counts));
    return Status::OK();
  }

  Status Combine(const std::shared_ptr<arrow::Array> &value_col, const int64_t *local_group_ids,
                 int64_t local_num_groups, arrow::ArrayVector *combined_results) const override {
    assert(combined_results->size() == num_arrays());
    assert(local_num_groups == (*combined_results)[0]->length());

    T *sums = (*combined_results)[0]->data()->template GetMutableValues<T>(1);
    auto *counts = (*combined_results)[1]->data()->template GetMutableValues<int64_t>(1);

    return CombineVisit<ArrowT>(value_col, local_group_ids,
                                [&](const T &val, int64_t gid) {
                                  sums[gid] += val;
                                  counts[gid]++;
                                });
  }

  Status Reduce(const arrow::ArrayVector &combined_results, const int64_t *local_group_ids,
                const int64_t *local_group_indices, int64_t local_num_groups,
                arrow::ArrayVector *reduced_results) const override {
    assert(reduced_results->size() == num_arrays());
    assert(combined_results.size() == num_arrays());
    assert(local_num_groups == (*reduced_results)[0]->length());
    CYLON_UNUSED(local_group_indices);

    auto *sums = (*reduced_results)[0]->data()->template GetMutableValues<T>(1);

    RETURN_CYLON_STATUS_IF_FAILED(
        CombineVisit<ArrowT>(combined_results[0], local_group_ids,
                             [&](const T &val, int64_t gid) { sums[gid] += val; }));

    auto *counts = (*reduced_results)[1]->data()->template GetMutableValues<int64_t>(1);
    RETURN_CYLON_STATUS_IF_FAILED(
        CombineVisit<arrow::Int64Type>(combined_results[1], local_group_ids,
                                       [&](const T &val, int64_t gid) { counts[gid] += val; }));
    return Status::OK();
  }

  Status Finalize(const arrow::ArrayVector &combined_results,
                  std::shared_ptr<arrow::Array> *output) const override {
    auto *sums = combined_results[0]->data()->template GetValues<T>(1);
    auto *counts = combined_results[1]->data()->template GetValues<int64_t>(1);
    int64_t groups = combined_results[0]->length();

    return FinalizeVisit<ResultT>(pool, groups, output,
                                  [&](int64_t i) { return sums[i] / counts[i]; });
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

template<typename T>
std::unique_ptr<MapReduceKernel> MakeMapReduceKernelImpl(compute::AggregationOpId reduce_op) {
  switch (reduce_op) {
    case compute::SUM: return std::make_unique<SumKernelImpl<T>>();
    case compute::MIN:return std::make_unique<MinKernelImpl<T>>();
    case compute::MAX:return std::make_unique<MaxKernelImpl<T>>();
    case compute::COUNT:return std::make_unique<CountKernelImpl>();
    case compute::MEAN:return std::make_unique<MeanKernelImpl<T>>();
    case compute::VAR:break;
    case compute::NUNIQUE:break;
    case compute::QUANTILE:break;
    case compute::STDDEV:break;
  }
  return nullptr;
}

std::unique_ptr<MapReduceKernel> MakeMapReduceKernel(const std::shared_ptr<arrow::DataType> &type,
                                                     compute::AggregationOpId reduce_op) {
  switch (type->id()) {
    case arrow::Type::NA:break;
    case arrow::Type::BOOL:break;
    case arrow::Type::UINT8: return MakeMapReduceKernelImpl<arrow::UInt8Type>(reduce_op);
    case arrow::Type::INT8:return MakeMapReduceKernelImpl<arrow::Int8Type>(reduce_op);
    case arrow::Type::UINT16:return MakeMapReduceKernelImpl<arrow::UInt16Type>(reduce_op);
    case arrow::Type::INT16:return MakeMapReduceKernelImpl<arrow::Int16Type>(reduce_op);
    case arrow::Type::UINT32:return MakeMapReduceKernelImpl<arrow::UInt32Type>(reduce_op);
    case arrow::Type::INT32:return MakeMapReduceKernelImpl<arrow::Int32Type>(reduce_op);
    case arrow::Type::UINT64:return MakeMapReduceKernelImpl<arrow::UInt64Type>(reduce_op);
    case arrow::Type::INT64:return MakeMapReduceKernelImpl<arrow::Int64Type>(reduce_op);
    case arrow::Type::HALF_FLOAT:break;
    case arrow::Type::FLOAT:return MakeMapReduceKernelImpl<arrow::FloatType>(reduce_op);;
    case arrow::Type::DOUBLE:return MakeMapReduceKernelImpl<arrow::DoubleType>(reduce_op);;
    case arrow::Type::STRING:break;
    case arrow::Type::BINARY:break;
    case arrow::Type::FIXED_SIZE_BINARY:break;
    case arrow::Type::DATE32:break;
    case arrow::Type::DATE64:break;
    case arrow::Type::TIMESTAMP:break;
    case arrow::Type::TIME32:break;
    case arrow::Type::TIME64:break;
    case arrow::Type::INTERVAL_MONTHS:break;
    case arrow::Type::INTERVAL_DAY_TIME:break;
    case arrow::Type::DECIMAL128:break;
    case arrow::Type::DECIMAL256:break;
    case arrow::Type::LIST:break;
    case arrow::Type::STRUCT:break;
    case arrow::Type::SPARSE_UNION:break;
    case arrow::Type::DENSE_UNION:break;
    case arrow::Type::DICTIONARY:break;
    case arrow::Type::MAP:break;
    case arrow::Type::EXTENSION:break;
    case arrow::Type::FIXED_SIZE_LIST:break;
    case arrow::Type::DURATION:break;
    case arrow::Type::LARGE_STRING:break;
    case arrow::Type::LARGE_BINARY:break;
    case arrow::Type::LARGE_LIST:break;
    case arrow::Type::MAX_ID:break;
  }
  return nullptr;
}

/**
 * 1. map keys to groups in the local table
 * 2. combine locally
 * 3. shuffle combined results
 * 4. map keys to groups in the shuffled table
 * 5. reduce shuffled table locally
 * 6. finalize reduction
 */
Status Aggregate(const std::shared_ptr<CylonContext> &ctx,
                 const std::shared_ptr<Table> &table,
                 const std::vector<int> &key_cols,
                 const std::vector<std::pair<int, compute::AggregationOpId>> &aggs,
                 std::shared_ptr<Table> *output,
                 const MapToGroupKernel &mapper) {



  return Status::OK();
}
}
}
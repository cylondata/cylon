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
void CombineVisit(const std::shared_ptr<arrow::Array> &value_col, const int64_t *local_group_ids,
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
}

Status AllocateArray(arrow::MemoryPool *pool, const std::shared_ptr<arrow::DataType> &type,
                     int64_t length, std::shared_ptr<arrow::Array> *array) {
  std::unique_ptr<arrow::ArrayBuilder> builder;
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(arrow::MakeBuilder(pool, type, &builder));
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder->AppendEmptyValues(length));
  CYLON_ASSIGN_OR_RAISE(*array, builder->Finish())
  return Status::OK();
}

Status AllocateArrays(arrow::MemoryPool *pool, const arrow::DataTypeVector &intermediate_types,
                      int64_t length, arrow::ArrayVector *arrays) {
  arrays->clear();
  arrays->reserve(intermediate_types.size());
  for (const auto &type: intermediate_types) {
    std::shared_ptr<arrow::Array> arr;
    RETURN_CYLON_STATUS_IF_FAILED(AllocateArray(pool, type, length, &arr));
    arrays->push_back(std::move(arr));
  }
  return Status::OK();
}

size_t MapReduceKernel::num_arrays() const { return intermediate_types().size(); }

template<typename ArrowT, template<typename> class CombineOp>
struct MapReduceKernelImpl1D : public MapReduceKernel {
  using T = typename ArrowT::c_type;

 public:
  explicit MapReduceKernelImpl1D(const std::shared_ptr<arrow::DataType> &in_type)
      : out_types({in_type}) {}

  const std::shared_ptr<arrow::DataType> &output_type() const override { return out_types[0]; }
  const arrow::DataTypeVector &intermediate_types() const override { return out_types; }

  void Init(arrow::MemoryPool *pool, compute::KernelOptions *options) override {
    CYLON_UNUSED(options);
    this->pool_ = pool;
  }

  Status CombineLocally(const std::shared_ptr<arrow::Array> &value_col,
                        const std::shared_ptr<arrow::Array> &local_group_ids,
                        int64_t local_num_groups,
                        arrow::ArrayVector *combined_results) const override {
    std::shared_ptr<arrow::Array> arr;
    RETURN_CYLON_STATUS_IF_FAILED(AllocateArray(pool_, out_types[0], local_num_groups, &arr));

    auto *g_ids = local_group_ids->data()->template GetValues<int64_t>(1);
    T *res = arr->data()->template GetMutableValues<T>(1);
    CombineVisit<ArrowT>(value_col, g_ids,
                         [&](const T &val, int64_t gid) {
                           res[gid] = CombineOp<T>::Call(res[gid], val);
                         });

    *combined_results = {std::move(arr)};
    return Status::OK();
  }

  Status ReduceShuffledResults(const arrow::ArrayVector &combined_results,
                               const std::shared_ptr<arrow::Array> &local_group_ids,
                               const std::shared_ptr<arrow::Array> &local_group_indices,
                               int64_t local_num_groups,
                               arrow::ArrayVector *reduced_results) const override {
    CYLON_UNUSED(local_group_indices);
    return CombineLocally(combined_results[0], local_group_ids,
                          local_num_groups, reduced_results);
  }

  Status Finalize(const arrow::ArrayVector &combined_results,
                  std::shared_ptr<arrow::Array> *output) const override {
    *output = combined_results[0];
    return Status::OK();
  }

 protected:
  const arrow::DataTypeVector out_types;
  arrow::MemoryPool *pool_ = nullptr;
};

template<typename T>
struct SumFunction {
  static T Call(const T &x, const T &y) { return x + y; };
};
template<typename ArrowT>
struct SumKernelImpl : public MapReduceKernelImpl1D<ArrowT, SumFunction> {
  explicit SumKernelImpl(const std::shared_ptr<arrow::DataType> &type)
      : MapReduceKernelImpl1D<ArrowT, SumFunction>(type) {}
  std::string name() const override { return "sum"; }
};

template<typename T>
struct MinFunction {
  static T Call(const T &x, const T &y) { return std::min(x, y); };
};
template<typename ArrowT>
struct MinKernelImpl : public MapReduceKernelImpl1D<ArrowT, MinFunction> {
  explicit MinKernelImpl(const std::shared_ptr<arrow::DataType> &type)
      : MapReduceKernelImpl1D<ArrowT, MinFunction>(type) {}
  std::string name() const override { return "min"; }
};

template<typename T>
struct MaxFunction {
  static T Call(const T &x, const T &y) { return std::max(x, y); };
};
template<typename ArrowT>
struct MaxKernelImpl : public MapReduceKernelImpl1D<ArrowT, MaxFunction> {
  explicit MaxKernelImpl(const std::shared_ptr<arrow::DataType> &type)
      : MapReduceKernelImpl1D<ArrowT, MaxFunction>(type) {}
  std::string name() const override { return "max"; }
};

struct CountKernelImpl : public MapReduceKernel {
  const arrow::DataTypeVector out_types{arrow::int64()};
  arrow::MemoryPool *pool_ = nullptr;

  std::string name() const override { return "count"; }
  const std::shared_ptr<arrow::DataType> &output_type() const override { return out_types[0]; }
  const arrow::DataTypeVector &intermediate_types() const override { return out_types; }

  void Init(arrow::MemoryPool *pool, compute::KernelOptions *options) override {
    CYLON_UNUSED(options);
    this->pool_ = pool;
  }

  Status CombineLocally(const std::shared_ptr<arrow::Array> &value_col,
                        const std::shared_ptr<arrow::Array> &local_group_ids,
                        int64_t local_num_groups,
                        arrow::ArrayVector *combined_results) const override {
    std::shared_ptr<arrow::Array> arr;
    RETURN_CYLON_STATUS_IF_FAILED(AllocateArray(pool_, out_types[0], local_num_groups, &arr));

    auto *g_ids = local_group_ids->data()->template GetValues<int64_t>(1);
    auto *counts = arr->data()->template GetMutableValues<int64_t>(1);
    for (int64_t i = 0; i < value_col->length(); i++) {
      counts[g_ids[i]]++;
    }

    *combined_results = {std::move(arr)};
    return Status::OK();
  }

  Status ReduceShuffledResults(const arrow::ArrayVector &combined_results,
                               const std::shared_ptr<arrow::Array> &local_group_ids,
                               const std::shared_ptr<arrow::Array> &local_group_indices,
                               int64_t local_num_groups,
                               arrow::ArrayVector *reduced_results) const override {
    CYLON_UNUSED(local_group_indices);
    std::shared_ptr<arrow::Array> arr;
    RETURN_CYLON_STATUS_IF_FAILED(AllocateArray(pool_, out_types[0], local_num_groups, &arr));

    auto *g_ids = local_group_ids->data()->template GetValues<int64_t>(1);
    auto *res = arr->data()->template GetMutableValues<int64_t>(1);
    CombineVisit<arrow::Int64Type>(combined_results[0], g_ids,
                                   [&](const int64_t &val, int64_t gid) {
                                     res[gid] += val;
                                   });
    *reduced_results = {std::move(arr)};
    return Status::OK();
  }

  Status Finalize(const arrow::ArrayVector &combined_results,
                  std::shared_ptr<arrow::Array> *output) const override {
    *output = combined_results[0];
    return Status::OK();
  }
};

template<typename ArrowT>
struct MeanKernelImpl : public MapReduceKernel {
  using T = typename ArrowT::c_type;

 public:
  explicit MeanKernelImpl(const std::shared_ptr<arrow::DataType> &in_type)
      : inter_types({in_type, arrow::int64()}) {}

  std::string name() const override { return "mean"; }
  const std::shared_ptr<arrow::DataType> &output_type() const override { return inter_types[0]; }
  const arrow::DataTypeVector &intermediate_types() const override { return inter_types; }

  void Init(arrow::MemoryPool *pool, compute::KernelOptions *options) override {
    CYLON_UNUSED(options);
    this->pool_ = pool;
  }

  Status CombineLocally(const std::shared_ptr<arrow::Array> &value_col,
                        const std::shared_ptr<arrow::Array> &local_group_ids,
                        int64_t local_num_groups,
                        arrow::ArrayVector *combined_results) const override {
    RETURN_CYLON_STATUS_IF_FAILED(AllocateArrays(pool_, inter_types, local_num_groups,
                                                 combined_results));

    auto *g_ids = local_group_ids->data()->template GetValues<int64_t>(1);
    T *sums = (*combined_results)[0]->data()->template GetMutableValues<T>(1);
    auto *counts = (*combined_results)[1]->data()->template GetMutableValues<int64_t>(1);

    CombineVisit<ArrowT>(value_col, g_ids,
                         [&](const T &val, int64_t gid) {
                           sums[gid] += val;
                           counts[gid]++;
                         });
    return Status::OK();
  }

  Status ReduceShuffledResults(const arrow::ArrayVector &combined_results,
                               const std::shared_ptr<arrow::Array> &local_group_ids,
                               const std::shared_ptr<arrow::Array> &local_group_indices,
                               int64_t local_num_groups,
                               arrow::ArrayVector *reduced_results) const override {
    assert(combined_results.size() == num_arrays());
    CYLON_UNUSED(local_group_indices);
    auto *g_ids = local_group_ids->data()->template GetValues<int64_t>(1);

    std::shared_ptr<arrow::Array> sum_arr;
    RETURN_CYLON_STATUS_IF_FAILED(AllocateArray(pool_, inter_types[0], local_num_groups, &sum_arr));
    auto *sums = sum_arr->data()->template GetMutableValues<T>(1);
    CombineVisit<ArrowT>(combined_results[0], g_ids,
                         [&](const T &val, int64_t gid) { sums[gid] += val; });

    std::shared_ptr<arrow::Array> cnt_arr;
    RETURN_CYLON_STATUS_IF_FAILED(AllocateArray(pool_, inter_types[1], local_num_groups, &cnt_arr));
    auto *counts = cnt_arr->data()->template GetMutableValues<int64_t>(1);
    CombineVisit<arrow::Int64Type>(combined_results[1], g_ids,
                                   [&](const T &val, int64_t gid) { counts[gid] += val; });

    *reduced_results = {std::move(sum_arr), std::move(cnt_arr)};
    return Status::OK();
  }

  Status Finalize(const arrow::ArrayVector &combined_results,
                  std::shared_ptr<arrow::Array> *output) const override {
    assert(combined_results.size() == num_arrays());

    int64_t num_groups = combined_results[0]->length();
    assert(combined_results[1]->length() == num_groups);

    auto *sums = combined_results[0]->data()->template GetMutableValues<T>(1);
    const auto *counts = combined_results[1]->data()->template GetValues<int64_t>(1);

    // now make the mean
    for (int64_t i = 0; i < num_groups; i++) {
      sums[i] = static_cast<T>(sums[i] / counts[i]);
    }

    *output = combined_results[0];
    return Status::OK();
  }

 private:
  arrow::MemoryPool *pool_ = nullptr;
  // {sum_t, count_t}
  const arrow::DataTypeVector inter_types;
};

template<typename ArrowT, bool StdDev = false>
struct VarKernelImpl : public MapReduceKernel {
  using T = typename ArrowT::c_type;

  arrow::MemoryPool *pool_ = nullptr;
  int ddof = 0;
  // {sum_sq_t, sum_t, count_t}
  const arrow::DataTypeVector inter_types{arrow::float64(), arrow::float64(), arrow::int64()};

  std::string name() const override { return StdDev ? "std" : "mean"; }
  const std::shared_ptr<arrow::DataType> &output_type() const override { return inter_types[0]; }
  const arrow::DataTypeVector &intermediate_types() const override { return inter_types; }

  void Init(arrow::MemoryPool *pool, compute::KernelOptions *options) override {
    this->ddof = reinterpret_cast<compute::VarKernelOptions *>(options)->ddof;
    this->pool_ = pool;
  }

  Status CombineLocally(const std::shared_ptr<arrow::Array> &value_col,
                        const std::shared_ptr<arrow::Array> &local_group_ids,
                        int64_t local_num_groups,
                        arrow::ArrayVector *combined_results) const override {
    RETURN_CYLON_STATUS_IF_FAILED(AllocateArrays(pool_, inter_types, local_num_groups,
                                                 combined_results));
    auto *g_ids = local_group_ids->data()->template GetValues<int64_t>(1);
    auto *sq_sums = (*combined_results)[0]->data()->template GetMutableValues<double>(1);
    auto *sums = (*combined_results)[1]->data()->template GetMutableValues<double>(1);
    auto *counts = (*combined_results)[2]->data()->template GetMutableValues<int64_t>(1);

    CombineVisit<ArrowT>(value_col, g_ids,
                         [&](const T &val, int64_t gid) {
                           sq_sums[gid] += (double) (val * val);
                           sums[gid] += (double) (val);
                           counts[gid]++;
                         });
    return Status::OK();
  }

  Status ReduceShuffledResults(const arrow::ArrayVector &combined_results,
                               const std::shared_ptr<arrow::Array> &local_group_ids,
                               const std::shared_ptr<arrow::Array> &local_group_indices,
                               int64_t local_num_groups,
                               arrow::ArrayVector *reduced_results) const override {
    assert(combined_results.size() == num_arrays());
    assert(combined_results[0]->type() == inter_types[0]);
    assert(combined_results[1]->type() == inter_types[1]);
    assert(combined_results[2]->type() == inter_types[2]);

    CYLON_UNUSED(local_group_indices);
    auto *g_ids = local_group_ids->data()->template GetValues<int64_t>(1);

    std::shared_ptr<arrow::Array> sq_sum_arr;
    RETURN_CYLON_STATUS_IF_FAILED(AllocateArray(pool_, inter_types[0], local_num_groups,
                                                &sq_sum_arr));
    auto *sq_sums = sq_sum_arr->data()->template GetMutableValues<double>(1);
    CombineVisit<arrow::DoubleType>(combined_results[0], g_ids,
                                    [&](const double &val, int64_t gid) { sq_sums[gid] += val; });

    std::shared_ptr<arrow::Array> sum_arr;
    RETURN_CYLON_STATUS_IF_FAILED(AllocateArray(pool_, inter_types[1], local_num_groups, &sum_arr));
    auto *sums = sum_arr->data()->template GetMutableValues<double>(1);
    CombineVisit<arrow::DoubleType>(combined_results[1], g_ids,
                                    [&](const double &val, int64_t gid) { sums[gid] += val; });

    std::shared_ptr<arrow::Array> cnt_arr;
    RETURN_CYLON_STATUS_IF_FAILED(AllocateArray(pool_, inter_types[2], local_num_groups, &cnt_arr));
    auto *counts = cnt_arr->data()->template GetMutableValues<int64_t>(1);
    CombineVisit<arrow::Int64Type>(combined_results[2], g_ids,
                                   [&](const int64_t &val, int64_t gid) { counts[gid] += val; });

    *reduced_results = {std::move(sq_sum_arr), std::move(sum_arr), std::move(cnt_arr)};
    return Status::OK();
  }

  Status Finalize(const arrow::ArrayVector &combined_results,
                  std::shared_ptr<arrow::Array> *output) const override {
    assert(combined_results.size() == num_arrays());

    int64_t num_groups = combined_results[0]->length();
    assert(combined_results[1]->length() == num_groups);
    assert(combined_results[2]->length() == num_groups);

    auto *sq_sums = combined_results[0]->data()->template GetMutableValues<double>(1);
    auto *sums = combined_results[1]->data()->template GetValues<double>(1);
    auto *counts = combined_results[2]->data()->template GetValues<int64_t>(1);

    // now make the mean
    int64_t i = 0;
    for (; i < num_groups; i++) {
      auto count = static_cast<double>(counts[i]);
      double sq_sum = sq_sums[i];
      double sum = sums[i];
      if (count > 0) {
        double mean = sum / count;
        double mean_sum_sq = sq_sum / count;
        double var = sq_sum * (mean_sum_sq - mean * mean) / (count - ddof);
        sq_sums[i] = StdDev ? sqrt(var) : var;
      } else {
        break;
      }
    }

    if (i != num_groups) {
      return {Code::ExecutionError,
              "error occurred during var finalize. idx: " + std::to_string(i)};
    }
    *output = combined_results[0];
    return Status::OK();
  }
};

// todo: to be supported with arrow v6.0+
/*struct NuniqueKernelImpl : public MapReduceKernel {
  const arrow::DataTypeVector out_types{arrow::int64()};
  arrow::MemoryPool *pool_ = nullptr;

  std::string name() const override { return "nunique"; }
  const std::shared_ptr<arrow::DataType> &output_type() const override { return out_types[0]; }
  const arrow::DataTypeVector &intermediate_types() const override { return out_types; }

  void Init(arrow::MemoryPool *pool, compute::KernelOptions *options) override {
    CYLON_UNUSED(options);
    this->pool_ = pool;
  }

  Status CombineLocally(const std::shared_ptr<arrow::Array> &value_col,
                        const std::shared_ptr<arrow::Array> &local_group_ids,
                        int64_t local_num_groups,
                        arrow::ArrayVector *combined_results) const override {
    CYLON_UNUSED(local_group_ids);
    CYLON_UNUSED(local_num_groups);
    *combined_results = {value_col};
    return Status::OK();
  }

  Status ReduceShuffledResults(const arrow::ArrayVector &combined_results,
                               const std::shared_ptr<arrow::Array> &local_group_ids,
                               const std::shared_ptr<arrow::Array> &local_group_indices,
                               int64_t local_num_groups,
                               arrow::ArrayVector *reduced_results) const override {
    CYLON_UNUSED(combined_results);
    CYLON_UNUSED(local_group_ids);
    CYLON_UNUSED(local_group_indices);
    CYLON_UNUSED(local_num_groups);
    CYLON_UNUSED(reduced_results);
    return {Code::ExecutionError, "Nunique does not support ReduceShuffledResults"};
  }

  Status Finalize(const arrow::ArrayVector &combined_results,
                  std::shared_ptr<arrow::Array> *output) const override {
    *output = combined_results[0];
    return Status::OK();
  }
};*/

Status MapToGroupKernel::Map(const arrow::ArrayVector &arrays,
                             std::shared_ptr<arrow::Array> *local_group_ids,
                             std::shared_ptr<arrow::Array> *local_group_indices,
                             int64_t *local_num_groups) const {
  const int64_t num_rows = arrays[0]->length();
  if (std::any_of(arrays.begin() + 1, arrays.end(),
                  [&](const auto &arr) { return arr->length() != num_rows; })) {
    return {Code::Invalid, "array lengths should be the same"};
  }

  // if empty, return
  if (num_rows == 0) {
    CYLON_ASSIGN_OR_RAISE(*local_group_ids, arrow::MakeArrayOfNull(arrow::int64(), 0, pool_))
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

  arrow::Int64Builder group_ids_build(pool_);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(group_ids_build.Reserve(num_rows));

  arrow::Int64Builder filter_build(pool_);
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

Status MapToGroupKernel::Map(const std::shared_ptr<arrow::Table> &table,
                             const std::vector<int> &key_cols,
                             std::shared_ptr<arrow::Array> *local_group_ids,
                             std::shared_ptr<arrow::Array> *local_group_indices,
                             int64_t *local_num_groups) const {
  arrow::ArrayVector arrays;
  arrays.reserve(key_cols.size());

  for (int i: key_cols) {
    const auto &col = table->column(i);
    if (col->num_chunks() > 1) {
      return {Code::Invalid, "MapToGroupKernel doesnt support chunks"};
    }
    arrays.push_back(col->chunk(0));
  }

  return Map(arrays, local_group_ids, local_group_indices, local_num_groups);
}

template<typename T>
std::unique_ptr<MapReduceKernel> MakeMapReduceKernelImpl(
    const std::shared_ptr<arrow::DataType> &type, compute::AggregationOpId reduce_op) {
  switch (reduce_op) {
    case compute::SUM: return std::make_unique<SumKernelImpl<T>>(type);
    case compute::MIN:return std::make_unique<MinKernelImpl<T>>(type);
    case compute::MAX:return std::make_unique<MaxKernelImpl<T>>(type);
    case compute::COUNT:return std::make_unique<CountKernelImpl>();
    case compute::MEAN:return std::make_unique<MeanKernelImpl<T>>(type);
    case compute::VAR: return std::make_unique<VarKernelImpl<T>>();
    case compute::STDDEV:return std::make_unique<VarKernelImpl<T, true>>();
    case compute::NUNIQUE:break;
    case compute::QUANTILE:break;
  }
  return nullptr;
}

std::unique_ptr<MapReduceKernel> MakeMapReduceKernel(const std::shared_ptr<arrow::DataType> &type,
                                                     compute::AggregationOpId reduce_op) {
  switch (type->id()) {
    case arrow::Type::NA:break;
    case arrow::Type::BOOL:break;
    case arrow::Type::UINT8: return MakeMapReduceKernelImpl<arrow::UInt8Type>(type, reduce_op);
    case arrow::Type::INT8:return MakeMapReduceKernelImpl<arrow::Int8Type>(type, reduce_op);
    case arrow::Type::UINT16:return MakeMapReduceKernelImpl<arrow::UInt16Type>(type, reduce_op);
    case arrow::Type::INT16:return MakeMapReduceKernelImpl<arrow::Int16Type>(type, reduce_op);
    case arrow::Type::UINT32:return MakeMapReduceKernelImpl<arrow::UInt32Type>(type, reduce_op);
    case arrow::Type::INT32:return MakeMapReduceKernelImpl<arrow::Int32Type>(type, reduce_op);
    case arrow::Type::UINT64:return MakeMapReduceKernelImpl<arrow::UInt64Type>(type, reduce_op);
    case arrow::Type::INT64:return MakeMapReduceKernelImpl<arrow::Int64Type>(type, reduce_op);
    case arrow::Type::FLOAT:return MakeMapReduceKernelImpl<arrow::FloatType>(type, reduce_op);;
    case arrow::Type::DOUBLE:return MakeMapReduceKernelImpl<arrow::DoubleType>(type, reduce_op);;
    case arrow::Type::DATE32:return MakeMapReduceKernelImpl<arrow::Date32Type>(type, reduce_op);
    case arrow::Type::DATE64:return MakeMapReduceKernelImpl<arrow::Date64Type>(type, reduce_op);
    case arrow::Type::TIMESTAMP:
      return MakeMapReduceKernelImpl<arrow::TimestampType>(type,
                                                           reduce_op);
    case arrow::Type::TIME32:return MakeMapReduceKernelImpl<arrow::Time32Type>(type, reduce_op);
    case arrow::Type::TIME64:return MakeMapReduceKernelImpl<arrow::Time64Type>(type, reduce_op);
    case arrow::Type::HALF_FLOAT:break;
    case arrow::Type::STRING:break;
    case arrow::Type::LARGE_STRING:break;
    case arrow::Type::BINARY:break;
    case arrow::Type::LARGE_BINARY:break;
    case arrow::Type::FIXED_SIZE_BINARY:break;
    default: break;
  }
  return nullptr;
}

using AggKernelVector = std::vector<std::pair<int, std::shared_ptr<MapReduceKernel>>>;

Status MakeAggKernels(arrow::MemoryPool *pool,
                      const std::shared_ptr<arrow::Schema> &schema,
                      const AggOpVector &aggs,
                      AggKernelVector *agg_kernels) {
  agg_kernels->reserve(aggs.size());
  for (const auto &agg: aggs) {
    const auto &type = schema->field(agg.first)->type();
    auto kern = MakeMapReduceKernel(type, agg.second->id());

    if (kern) {
      // initialize kernel
      kern->Init(pool, agg.second->options());
      agg_kernels->emplace_back(agg.first, std::move(kern));
    } else {
      return {Code::NotImplemented, "Unsupported reduce kernel type " + type->ToString()};
    }
  }
  return Status::OK();
}

Status MakeOutputSchema(const std::shared_ptr<arrow::Schema> &cur_schema,
                        const std::vector<int> &key_cols, const AggKernelVector &agg_kernels,
                        std::shared_ptr<arrow::Schema> *out_schema,
                        bool use_intermediate_types = false) {
  arrow::SchemaBuilder schema_builder;
  // add key fields
  for (int i: key_cols) {
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(schema_builder.AddField(cur_schema->field(i)));
  }

  // add agg fields --> out field name = "name" + "op_name"
  for (const auto &agg: agg_kernels) {
    const auto &kern = agg.second;
    if (use_intermediate_types) {
      const auto &types = kern->intermediate_types();
      for (size_t i = 0; i < types.size(); i++) {
        auto f = arrow::field(
            cur_schema->field(agg.first)->name() + "_" + kern->name() + "_" + std::to_string(i),
            types[i]);
        RETURN_CYLON_STATUS_IF_ARROW_FAILED(schema_builder.AddField(f));
      }
    } else { // using output types
      auto f = arrow::field(cur_schema->field(agg.first)->name() + "_" + kern->name(),
                            kern->output_type());
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(schema_builder.AddField(f));
    }
  }

  CYLON_ASSIGN_OR_RAISE(*out_schema, schema_builder.Finish())
  return Status::OK();
}

Status MapToGroups(arrow::compute::ExecContext *exec_ctx,
                   const std::shared_ptr<arrow::Table> &atable,
                   const std::vector<int> &key_cols,
                   const MapToGroupKernel *mapper,
                   std::shared_ptr<arrow::Array> *group_ids,
                   std::shared_ptr<arrow::Array> *group_indices,
                   int64_t *num_groups,
                   arrow::ChunkedArrayVector *out_arrays) {
  RETURN_CYLON_STATUS_IF_FAILED(mapper->Map(atable, key_cols, group_ids, group_indices,
                                            num_groups));
  assert(*num_groups == (*group_indices)->length());
  assert(atable->num_rows() == (*group_ids)->length());

  // take group_indices from the columns to create out key columns
  for (int k: key_cols) {
    CYLON_ASSIGN_OR_RAISE(auto arr,
                          arrow::compute::Take(atable->column(k)->chunk(0), *group_indices,
                                               arrow::compute::TakeOptions::NoBoundsCheck(),
                                               exec_ctx))
    out_arrays->push_back(std::make_shared<arrow::ChunkedArray>(arr.make_array()));
  }

  return Status::OK();
}

Status LocalAggregate(const std::shared_ptr<CylonContext> &ctx,
                      const std::shared_ptr<arrow::Table> &atable,
                      const std::vector<int> &key_cols,
                      const AggKernelVector &agg_kernels,
                      std::shared_ptr<arrow::Table> *output,
                      const MapToGroupKernel *mapper) {
  auto pool = ToArrowPool(ctx);
  arrow::compute::ExecContext exec_ctx(pool);
  const auto &cur_schema = atable->schema();

  // make out_schema
  std::shared_ptr<arrow::Schema> out_schema;
  RETURN_CYLON_STATUS_IF_FAILED(MakeOutputSchema(cur_schema, key_cols, agg_kernels, &out_schema));

  if (atable->num_rows() == 0) {
    // return empty table with proper aggregate types
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(util::CreateEmptyTable(out_schema, output, pool));
    return Status::OK();
  }

  std::vector<std::shared_ptr<arrow::ChunkedArray>> out_arrays;
  out_arrays.reserve(out_schema->num_fields());

  // map to groups
  std::shared_ptr<arrow::Array> group_ids, group_indices;
  int64_t num_groups;

  RETURN_CYLON_STATUS_IF_FAILED(MapToGroups(&exec_ctx, atable, key_cols, mapper, &group_ids,
                                            &group_indices, &num_groups, &out_arrays));

  // make the value columns --> only use MapReduceKernel.CombineBeforeShuffle, and Finalize
  for (const auto &p: agg_kernels) {
    int val_col = p.first;
    const auto &kern = p.second;

    // val col
    if (atable->column(val_col)->num_chunks() > 1) {
      return {Code::Invalid, "Aggregates do not support chunks"};
    }
    const auto &val_arr = atable->column(val_col)->chunk(0);

    arrow::ArrayVector res;
    RETURN_CYLON_STATUS_IF_FAILED(
        kern->CombineLocally(val_arr, group_ids, num_groups, &res));
    std::shared_ptr<arrow::Array> out_arr;
    RETURN_CYLON_STATUS_IF_FAILED(kern->Finalize(res, &out_arr));

    out_arrays.push_back(std::make_shared<arrow::ChunkedArray>(std::move(out_arr)));
  }

  // check if the types match
  assert(out_arrays.size() == (size_t) out_schema->num_fields());
  for (int i = 0; i < out_schema->num_fields(); i++) {
    assert(out_schema->field(i)->type()->Equals(*out_arrays[i]->type()));
  }

  *output = arrow::Table::Make(std::move(out_schema), std::move(out_arrays));
  return Status::OK();
}

Status DistAggregate(const std::shared_ptr<CylonContext> &ctx,
                     const std::shared_ptr<arrow::Table> &atable,
                     const std::vector<int> &key_cols,
                     const AggKernelVector &agg_kernels,
                     std::shared_ptr<arrow::Table> *output,
                     const MapToGroupKernel *mapper) {
  auto pool = ToArrowPool(ctx);
  arrow::compute::ExecContext exec_ctx(pool);
  const auto &cur_schema = atable->schema();

  // make intermediate_schema
  std::shared_ptr<arrow::Schema> interim_schema;
  RETURN_CYLON_STATUS_IF_FAILED(
      MakeOutputSchema(cur_schema, key_cols, agg_kernels, &interim_schema, true));

  std::vector<std::shared_ptr<arrow::ChunkedArray>> interim_arrays;
  interim_arrays.reserve(interim_schema->num_fields());

  if (atable->num_rows() == 0) {
    // if table empty, push empty arrays to the interim_arrays
    for (const auto &f: interim_schema->fields()) {
      CYLON_ASSIGN_OR_RAISE(auto arr, arrow::MakeArrayOfNull(f->type(), 0, pool))
      interim_arrays.push_back(std::make_shared<arrow::ChunkedArray>(std::move(arr)));
    }
  } else {
    // map to groups
    std::shared_ptr<arrow::Array> group_ids, group_indices;
    int64_t num_groups;
    RETURN_CYLON_STATUS_IF_FAILED(MapToGroups(&exec_ctx, atable, key_cols, mapper, &group_ids,
                                              &group_indices, &num_groups, &interim_arrays));

    // make the interim value columns --> only use MapReduceKernel.CombineBeforeShuffle
    for (const auto &p: agg_kernels) {
      int val_col = p.first;
      const auto &kern = p.second;

      // val col
      if (atable->column(val_col)->num_chunks() > 1) {
        return {Code::Invalid, "Aggregates do not support chunks"};
      }
      const auto &val_arr = atable->column(val_col)->chunk(0);

      arrow::ArrayVector results;
      RETURN_CYLON_STATUS_IF_FAILED(
          kern->CombineLocally(val_arr, group_ids, num_groups, &results));

      for (auto &&arr: results) {
        interim_arrays.push_back(std::make_shared<arrow::ChunkedArray>(std::move(arr)));
      }
    }

    // check if the types match
    assert(interim_arrays.size() == (size_t) interim_schema->num_fields());
    for (int i = 0; i < interim_schema->num_fields(); i++) {
      assert(interim_schema->field(i)->type()->Equals(*interim_arrays[i]->type()));
    }

    // clear values
    group_ids.reset();
    group_indices.reset();
  }

  // now shuffle the interim results
  std::shared_ptr<Table> shuffle_table;
  RETURN_CYLON_STATUS_IF_FAILED(Table::FromArrowTable(ctx,
                                                      arrow::Table::Make(std::move(interim_schema),
                                                                         std::move(interim_arrays)),
                                                      shuffle_table));
  // key cols are at the front. So, create a new vector
  std::vector<int> shuffle_keys(key_cols.size());
  std::iota(shuffle_keys.begin(), shuffle_keys.end(), 0);
  // shuffle and update
  RETURN_CYLON_STATUS_IF_FAILED(Shuffle(shuffle_table, shuffle_keys, shuffle_table));
  const auto &shuffled_atable = shuffle_table->get_table();

  // make output schema
  std::shared_ptr<arrow::Schema> out_schema;
  RETURN_CYLON_STATUS_IF_FAILED(MakeOutputSchema(cur_schema, key_cols, agg_kernels, &out_schema));

  // now create new set of columns
  std::vector<std::shared_ptr<arrow::ChunkedArray>> out_arrays;
  out_arrays.reserve(out_schema->num_fields());

  if (shuffled_atable->num_rows() == 0) {
    // if table empty, push empty arrays to the out_arrays
    for (const auto &f: out_schema->fields()) {
      CYLON_ASSIGN_OR_RAISE(auto arr, arrow::MakeArrayOfNull(f->type(), 0, pool))
      out_arrays.push_back(std::make_shared<arrow::ChunkedArray>(std::move(arr)));
    }
  } else {
    // map to groups
    std::shared_ptr<arrow::Array> group_ids, group_indices;
    int64_t num_groups;
    RETURN_CYLON_STATUS_IF_FAILED(
        MapToGroups(&exec_ctx, shuffled_atable, shuffle_keys, mapper,
                    &group_ids, &group_indices, &num_groups, &out_arrays));

    // make the interim value columns --> only use MapReduceKernel.Init, ReduceAfterShuffle and Finalize
    size_t col_offset = shuffle_keys.size();
    for (const auto &p: agg_kernels) {
      const auto &kern = p.second;

      // recreate combined columns vector
      arrow::ArrayVector combined_results;
      combined_results.reserve(kern->num_arrays());
      for (size_t i = 0; i < kern->num_arrays(); i++) {
        assert (shuffled_atable->column(col_offset + i)->num_chunks() == 1);
        combined_results.push_back(shuffled_atable->column((int) (col_offset + i))->chunk(0));
      }
      col_offset += kern->num_arrays();

      arrow::ArrayVector results;
      RETURN_CYLON_STATUS_IF_FAILED(
          kern->ReduceShuffledResults(combined_results, group_ids, group_indices, num_groups,
                                      &results));
      std::shared_ptr<arrow::Array> out_arr;
      RETURN_CYLON_STATUS_IF_FAILED(kern->Finalize(results, &out_arr));

      out_arrays.push_back(std::make_shared<arrow::ChunkedArray>(std::move(out_arr)));
    }
  }

  // check if the types match
  assert(out_arrays.size() == (size_t) out_schema->num_fields());
  for (int i = 0; i < out_schema->num_fields(); i++) {
    assert(out_schema->field(i)->type()->Equals(*out_arrays[i]->type()));
  }
  *output = arrow::Table::Make(std::move(out_schema), std::move(out_arrays));
  return Status::OK();
}

// todo test this with new kernels
Status ShuffleTable(const std::shared_ptr<CylonContext> &ctx,
                    const std::shared_ptr<arrow::Table> &atable,
                    const std::vector<int> &key_cols, const AggKernelVector &agg_kernels,
                    std::shared_ptr<Table> *shuffled, std::vector<int> *new_key_cols,
                    AggKernelVector *new_agg_kernels) {
  arrow::FieldVector fields;
  arrow::ChunkedArrayVector arrays;

  // first take the key columns
  for (int k: key_cols) {
    arrays.push_back(atable->column(k));
    fields.push_back(atable->field(k));
  }

  new_key_cols->resize(key_cols.size());
  std::iota(new_key_cols->begin(), new_key_cols->end(), 0);

  new_agg_kernels->reserve(agg_kernels.size());

  std::unordered_map<int, int> val_cols; // original col_id -> new col_id
  for (const auto &agg: agg_kernels) {
    int original_col_id = agg.first;
    const auto &res = val_cols.emplace(original_col_id, key_cols.size() + val_cols.size());

    if (res.second) { // this is a unique col_id
      arrays.push_back(atable->column(original_col_id));
      fields.push_back(atable->field(original_col_id));
    }

    int new_col_id = res.first->second;
    new_agg_kernels->emplace_back(new_col_id, agg.second);
  }

  auto shuffling_atable = arrow::Table::Make(std::make_shared<arrow::Schema>(std::move(fields)),
                                             std::move(arrays));
  auto shuffling_table = std::make_shared<Table>(ctx, std::move(shuffling_atable));
  return Shuffle(shuffling_table, *new_key_cols, *shuffled);
}

// todo test this with new kernels
Status DistAggregateSingleStage(const std::shared_ptr<CylonContext> &ctx,
                                const std::shared_ptr<arrow::Table> &atable,
                                const std::vector<int> &key_cols,
                                const AggKernelVector &agg_kernels,
                                std::shared_ptr<arrow::Table> *output,
                                const MapToGroupKernel *mapper) {
  auto pool = ToArrowPool(ctx);
  arrow::compute::ExecContext exec_ctx(pool);

  std::vector<int> new_key_cols;
  AggKernelVector new_agg_kernels;
  std::shared_ptr<Table> shuffled_table;
  RETURN_CYLON_STATUS_IF_FAILED(ShuffleTable(ctx, atable, key_cols, agg_kernels,
                                             &shuffled_table, &new_key_cols, &new_agg_kernels));
  const auto &shuffled_atable = shuffled_table->get_table();
  const auto &shuffled_schema = atable->schema();

  // make output schema
  std::shared_ptr<arrow::Schema> out_schema;
  RETURN_CYLON_STATUS_IF_FAILED(MakeOutputSchema(shuffled_schema, new_key_cols, new_agg_kernels,
                                                 &out_schema));

  // now create new set of columns
  std::vector<std::shared_ptr<arrow::ChunkedArray>> out_arrays;
  out_arrays.reserve(out_schema->num_fields());

  if (shuffled_atable->num_rows() == 0) {
    // if table empty, push empty arrays to the out_arrays
    for (const auto &f: out_schema->fields()) {
      CYLON_ASSIGN_OR_RAISE(auto arr, arrow::MakeArrayOfNull(f->type(), 0, pool))
      out_arrays.push_back(std::make_shared<arrow::ChunkedArray>(std::move(arr)));
    }
  } else {
    // map to groups
    std::shared_ptr<arrow::Array> group_ids, group_indices;
    int64_t num_groups;
    RETURN_CYLON_STATUS_IF_FAILED(
        MapToGroups(&exec_ctx, shuffled_atable, new_key_cols, mapper, &group_ids, &group_indices,
                    &num_groups, &out_arrays));

    // make the interim value columns --> only use MapReduceKernel.CombineLocally and Finalize
    for (const auto &new_agg_kernel: new_agg_kernels) {
      int val_col = new_agg_kernel.first;
      const auto &kern = new_agg_kernel.second;

      assert(kern->num_arrays() == 1);
      const auto &val_arr = shuffled_atable->column(val_col)->chunk(0);

      arrow::ArrayVector results;
      RETURN_CYLON_STATUS_IF_FAILED(kern->CombineLocally(val_arr, group_ids, num_groups, &results));

      assert(results.size() == 1);
      std::shared_ptr<arrow::Array> out_arr;
      RETURN_CYLON_STATUS_IF_FAILED(kern->Finalize(results, &out_arr));

      out_arrays.push_back(std::make_shared<arrow::ChunkedArray>(std::move(out_arr)));
    }
  }

  // check if the types match
  assert(out_arrays.size() == (size_t) out_schema->num_fields());
  for (int i = 0; i < out_schema->num_fields(); i++) {
    assert(out_schema->field(i)->type()->Equals(*out_arrays[i]->type()));
  }
  *output = arrow::Table::Make(std::move(out_schema), std::move(out_arrays));
  return Status::OK();
}

Status MapredHashGroupBy(const std::shared_ptr<Table> &table, const std::vector<int> &key_cols,
                         const AggOpVector &aggs, std::shared_ptr<Table> *output,
                         const std::unique_ptr<MapToGroupKernel> &mapper) {
  const auto &ctx = table->GetContext();
  auto pool = ToArrowPool(ctx);
  const auto &atable = table->get_table();

  AggKernelVector agg_kernels;
  RETURN_CYLON_STATUS_IF_FAILED(MakeAggKernels(pool, atable->schema(), aggs, &agg_kernels));

  if (ctx->GetWorldSize() == 1) { // if serial execution, just perform local aggregate.
    std::shared_ptr<arrow::Table> out_table;
    RETURN_CYLON_STATUS_IF_FAILED(LocalAggregate(ctx, atable, key_cols, agg_kernels,
                                                 &out_table, mapper.get()));
    return Table::FromArrowTable(ctx, std::move(out_table), *output);
  }

  // distributed execution

  // there are 2 types of distributed kernels.
  // single stage - bypass local combine prior to the shuffle
  // dual stage - local combine + shuffle + local combine

  // push the single_stage_reduction kernels to the end of the vector
  auto single_stage_kernels_start
      = std::partition(agg_kernels.begin(), agg_kernels.end(),
                       [](const std::pair<int, std::shared_ptr<MapReduceKernel>> &k_pair) {
                         return !k_pair.second->single_stage_reduction();
                       });

  arrow::ChunkedArrayVector final_arrays;
  arrow::SchemaBuilder schema_builder;

  size_t dual_stage_kernels = std::distance(agg_kernels.begin(), single_stage_kernels_start);
  size_t single_stage_kernels = std::distance(single_stage_kernels_start, agg_kernels.end());

  if (dual_stage_kernels) { // i.e. there are some dual stage kernels
    std::shared_ptr<arrow::Table> temp_table;
    AggKernelVector kernels;
    std::move(agg_kernels.begin(), single_stage_kernels_start, std::back_inserter(kernels));

    RETURN_CYLON_STATUS_IF_FAILED(DistAggregate(ctx, atable, key_cols, kernels,
                                                &temp_table, mapper.get()));

    if (single_stage_kernels == 0) { // no single stage kernels, return temp_table
      return Table::FromArrowTable(ctx, std::move(temp_table), *output);
    } else {
      final_arrays = temp_table->columns(); // copy columns vector
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(schema_builder.AddSchema(temp_table->schema()));
    }
  }

  if (single_stage_kernels_start != agg_kernels.end()) {
    std::shared_ptr<arrow::Table> temp_table;
    AggKernelVector kernels;
    std::move(single_stage_kernels_start, agg_kernels.end(), std::back_inserter(kernels));
    RETURN_CYLON_STATUS_IF_FAILED(DistAggregateSingleStage(ctx, atable, key_cols, kernels,
                                                           &temp_table, mapper.get()));

    if (dual_stage_kernels == 0) { // there are only single_stage_kernels
      return Table::FromArrowTable(ctx, std::move(temp_table), *output);
    } else {
      assert(final_arrays[0]->length() == temp_table->num_rows());
      // skip key columns
      for (int i = (int) key_cols.size(); i < temp_table->num_columns(); i++) {
        final_arrays.push_back(temp_table->column(i));
        RETURN_CYLON_STATUS_IF_ARROW_FAILED(schema_builder.AddField(temp_table->field(i)));
      }
    }
  }

  CYLON_ASSIGN_OR_RAISE(auto schema, schema_builder.Finish())
  auto out_table = arrow::Table::Make(std::move(schema), std::move(final_arrays));
  return Table::FromArrowTable(ctx, std::move(out_table), *output);
}

Status MapredHashGroupBy(const std::shared_ptr<Table> &table, const std::vector<int> &key_cols,
                         const AggOpIdVector &aggs, std::shared_ptr<Table> *output) {
  AggOpVector op_vector;
  op_vector.reserve(aggs.size());

  for (auto p: aggs) {
    auto op = compute::MakeAggregationOpFromID(p.second);
    if (!op) {
      return {Code::Invalid, "Unable to create op for op_id " + std::to_string(p.second)};
    }
    op_vector.emplace_back(p.first, std::move(op));
  }
  return MapredHashGroupBy(table, key_cols, op_vector, output);
}

}
}

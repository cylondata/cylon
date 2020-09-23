//
// Created by niranda on 9/22/20.
//

#ifndef CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_OPS_HPP_
#define CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_OPS_HPP_

#include <functional>

#include "groupby_aggregate_ops.hpp"

namespace cylon {

template<typename T, cylon::GroupByAggregationOp AggregateOp>
struct AggregateKernel {

};

template<typename T>
struct AggregateKernel<T, cylon::GroupByAggregationOp::SUM> {
  using HashMapType = std::tuple<T>;
  using ResultType = T;

  static constexpr const char* _prefix = "sum_";

  static constexpr HashMapType Init(const T &value) {
    return HashMapType{value};
  }

  static inline void Update(const T &value, HashMapType *result) {
    std::get<0>(*result) += value;
  }

  static inline ResultType Finalize(const HashMapType *result) {
    return std::get<0>(*result);
  }
};

template<typename T>
struct AggregateKernel<T, cylon::GroupByAggregationOp::MIN> {
  using HashMapType = std::tuple<T>;
  using ResultType = T;
  static constexpr const char* _prefix = "min_";

  static constexpr HashMapType Init(const T &value) {
    return HashMapType{value};
  }

  static inline void Update(const T &value, HashMapType *result) {
    std::get<0>(*result) = std::min(value, std::get<0>(*result));
  }

  static inline ResultType Finalize(const HashMapType *result) {
    return std::get<0>(*result);
  }
};

template<typename T>
struct AggregateKernel<T, cylon::GroupByAggregationOp::MAX> {
  using HashMapType = std::tuple<T>;
  using ResultType = T;
  static constexpr const char* _prefix = "max_";

  static constexpr HashMapType Init(const T &value) {
    return HashMapType{value};
  }

  static inline void Update(const T &value, HashMapType *result) {
    std::get<0>(*result) = std::max(value, std::get<0>(*result));
  }

  static inline ResultType Finalize(const HashMapType *result) {
    return std::get<0>(*result);
  }
};

template<typename T>
struct AggregateKernel<T, GroupByAggregationOp::COUNT> {
  using HashMapType = std::tuple<int64_t>;
  using ResultType = int64_t;
  static constexpr const char* _prefix = "count_";

  static constexpr HashMapType Init(const T &value) {
    return HashMapType{1};
  }

  static inline void Update(const T &value, HashMapType *result) {
    std::get<0>(*result) += 1;
  }

  static inline ResultType Finalize(const HashMapType *result) {
    return std::get<0>(*result);
  }
};

template<typename T>
struct AggregateKernel<T, GroupByAggregationOp::MEAN> {
  using HashMapType = std::tuple<T, int64_t>;
  using ResultType = T;

  static constexpr char _prefix[] = "mean_";

  static constexpr HashMapType Init(const T &value) {
    return HashMapType{value, 1};
  }

  static inline void Update(const T &value, HashMapType *result) {
    std::get<0>(*result) += value;
    std::get<1>(*result) += 1;
  }

  static inline ResultType Finalize(const HashMapType *result) {
    return std::get<0>(*result) / std::get<1>(*result);
  }
};

template<typename IDX_T, typename VAL_T, cylon::GroupByAggregationOp AGG_OP,
    typename = typename std::enable_if<
        arrow::is_number_type<IDX_T>::value | arrow::is_boolean_type<IDX_T>::value>::type>
cylon::Status HashGroupBy(arrow::MemoryPool *pool,
                          const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                          const std::shared_ptr<arrow::ChunkedArray> &val_col,
                          std::vector<std::shared_ptr<arrow::Array>> &output_arrays) {
  // traverse idx and val arrays and create the hashmap
  using IDX_C_T = typename arrow::TypeTraits<IDX_T>::CType;
  using IDX_ARRAY_T = typename arrow::TypeTraits<IDX_T>::ArrayType;

  using VAL_C_T = typename arrow::TypeTraits<VAL_T>::CType;
  using VAL_ARRAY_T = typename arrow::TypeTraits<VAL_T>::ArrayType;

  using VAL_KERNEL = cylon::AggregateKernel<VAL_C_T, AGG_OP>;
  using VAL_HASHMAP_T = typename VAL_KERNEL::HashMapType;

  std::map<IDX_C_T, VAL_HASHMAP_T> hash_map;
//  hash_map.reserve(idx_col->length() * 0.5);

  const int chunks = idx_col->num_chunks();
  for (int chunk = 0; chunk < chunks; chunk++) {
    const int64_t len = idx_col->chunk(chunk)->length();
    const shared_ptr<IDX_ARRAY_T> &idx_arr = static_pointer_cast<IDX_ARRAY_T>(idx_col->chunk(chunk));
    const shared_ptr<VAL_ARRAY_T> &val_arr = static_pointer_cast<VAL_ARRAY_T>(val_col->chunk(chunk));

    IDX_C_T idx;
    VAL_C_T val;
    for (int i = 0; i < len; i++) {
      idx = idx_arr->Value(i);
      val = val_arr->Value(i);

      auto iter = hash_map.find(idx);

      if (iter == hash_map.end()) {
        hash_map.insert(std::make_pair(idx, VAL_KERNEL::Init(val)));
      } else {
        VAL_KERNEL::Update(val, &(iter->second));
      }
    }
  }

  // build arrow arrays
  arrow::Status s;
  if (output_arrays.empty()) { // if empty --> build the indx array
    using IDX_BUILDER_T = typename arrow::TypeTraits<IDX_T>::BuilderType;
    using VAL_BUILDER_T = typename arrow::TypeTraits<VAL_T>::BuilderType;

    IDX_BUILDER_T idx_builder(pool);
    VAL_BUILDER_T val_builder(pool);
    std::shared_ptr<arrow::Array> out_idx, out_val;

    const unsigned long groups = hash_map.size();

    if (!(s = idx_builder.Reserve(groups)).ok()) {
      return cylon::Status(static_cast<int>(s.code()), s.message());
    }

    if (!(s = val_builder.Reserve(groups)).ok()) {
      return cylon::Status(static_cast<int>(s.code()), s.message());
    }

    for (auto &p:  hash_map) {
      idx_builder.UnsafeAppend(p.first);
      val_builder.UnsafeAppend(VAL_KERNEL::Finalize(&(p.second)));
    }
    hash_map.clear();

    if (!(s = idx_builder.Finish(&out_idx)).ok()) {
      return cylon::Status(static_cast<int>(s.code()), s.message());
    }
    if (!(s = val_builder.Finish(&out_val)).ok()) {
      return cylon::Status(static_cast<int>(s.code()), s.message());
    }

    output_arrays.push_back(out_idx);
    output_arrays.push_back(out_val);

  } else { // only build the value array
    using VAL_BUILDER_T = typename arrow::TypeTraits<VAL_T>::BuilderType;

    VAL_BUILDER_T val_builder(pool);
    std::shared_ptr<arrow::Array> out_val;

    const unsigned long groups = hash_map.size();

    if (!(s = val_builder.Reserve(groups)).ok()) {
      return cylon::Status(static_cast<int>(s.code()), s.message());
    }

    for (auto &p:  hash_map) {
      val_builder.UnsafeAppend(VAL_KERNEL::Finalize(&(p.second)));
    }
    hash_map.clear();

    if (!(s = val_builder.Finish(&out_val)).ok()) {
      return cylon::Status(static_cast<int>(s.code()), s.message());
    }

    output_arrays.push_back(out_val);
  }

  return cylon::Status::OK();
}

typedef Status
(*HashGroupByFptr)(arrow::MemoryPool *pool,
                   const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                   const std::shared_ptr<arrow::ChunkedArray> &val_col,
                   std::vector<shared_ptr<arrow::Array>> &output_arrays);

template<typename IDX_T, typename VAL_T>
HashGroupByFptr ResolveOp(cylon::GroupByAggregationOp op) {
  switch (op) {
    case SUM: return &HashGroupBy<IDX_T, VAL_T, GroupByAggregationOp::SUM>();
    case COUNT: return &HashGroupBy<IDX_T, VAL_T, GroupByAggregationOp::COUNT>();
    case MIN:return &HashGroupBy<IDX_T, VAL_T, GroupByAggregationOp::MIN>();
    case MAX:return &HashGroupBy<IDX_T, VAL_T, GroupByAggregationOp::MAX>();
    case MEAN:return &HashGroupBy<IDX_T, VAL_T, GroupByAggregationOp::MEAN>();
  }
}

template<typename IDX_ARROW_T>
HashGroupByFptr PickHashGroupByFptr(const shared_ptr<cylon::DataType> &val_data_type,
                                    const cylon::GroupByAggregationOp op) {
  switch (val_data_type->getType()) {
    case Type::BOOL: return ResolveOp<IDX_ARROW_T, arrow::BooleanType>(op);
    case Type::UINT8: return ResolveOp<IDX_ARROW_T, arrow::UInt8Type>(op);
    case Type::INT8: return ResolveOp<IDX_ARROW_T, arrow::Int8Type>(op);
    case Type::UINT16: return ResolveOp<IDX_ARROW_T, arrow::UInt16Type>(op);
    case Type::INT16: return ResolveOp<IDX_ARROW_T, arrow::Int16Type>(op);
    case Type::UINT32: return ResolveOp<IDX_ARROW_T, arrow::UInt32Type>(op);
    case Type::INT32: return ResolveOp<IDX_ARROW_T, arrow::Int32Type>(op);
    case Type::UINT64: return ResolveOp<IDX_ARROW_T, arrow::UInt64Type>(op);
    case Type::INT64: return ResolveOp<IDX_ARROW_T, arrow::Int64Type>(op);
    case Type::FLOAT: return ResolveOp<IDX_ARROW_T, arrow::FloatType>(op);
    case Type::DOUBLE: return ResolveOp<IDX_ARROW_T, arrow::DoubleType>(op);
    case Type::HALF_FLOAT:break;
    case Type::STRING:break;
    case Type::BINARY:break;
    case Type::FIXED_SIZE_BINARY:break;
    case Type::DATE32:break;
    case Type::DATE64:break;
    case Type::TIMESTAMP:break;
    case Type::TIME32:break;
    case Type::TIME64:break;
    case Type::INTERVAL:break;
    case Type::DECIMAL:break;
    case Type::LIST:break;
    case Type::EXTENSION:break;
    case Type::FIXED_SIZE_LIST:break;
    case Type::DURATION:break;
  }
  return nullptr;
}

}

#endif //CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_OPS_HPP_

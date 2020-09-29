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

#ifndef CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_OPS_HPP_
#define CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_OPS_HPP_

#include <functional>

#include <arrow/api.h>
#include <data_types.hpp>
#include <table.hpp>
#include <ctx/arrow_memory_pool_utils.hpp>
#include "groupby_aggregate_ops.hpp"

namespace cylon {

template<typename T, cylon::GroupByAggregationOp AggregateOp,
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
struct AggregateKernel {

};

template<typename T>
struct AggregateKernel<T, cylon::GroupByAggregationOp::SUM> {
  using HashMapType = std::tuple<T>;
  using ResultType = T;
  using ResultArrowType = typename arrow::CTypeTraits<T>::ArrowType;

//  static constexpr const char* _prefix = "sum_";

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
  using ResultArrowType = typename arrow::CTypeTraits<T>::ArrowType;

//  static constexpr const char* _prefix = "min_";

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
  using ResultArrowType = typename arrow::CTypeTraits<T>::ArrowType;

//  static constexpr const char* _prefix = "max_";

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
  using ResultArrowType = arrow::Int64Type;

//  static constexpr const char* _prefix = "count_";

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

//template<typename T>
//struct AggregateKernel<T, GroupByAggregationOp::MEAN> {
//  using HashMapType = std::tuple<T, int64_t>;
//  using ResultType = T;
//  using ResultArrowType = typename arrow::CTypeTraits<T>::ArrowType;
//
////  static constexpr char _prefix[] = "mean_";
//
//  static constexpr HashMapType Init(const T &value) {
//    return HashMapType{value, 1};
//  }
//
//  static inline void Update(const T &value, HashMapType *result) {
//    std::get<0>(*result) += value;
//    std::get<1>(*result) += 1;
//  }
//
//  static inline ResultType Finalize(const HashMapType *result) {
//    return std::get<0>(*result) / std::get<1>(*result);
//  }
//};

template<typename IDX_T, typename VAL_T, cylon::GroupByAggregationOp AGG_OP,
    typename = typename std::enable_if<
        arrow::is_number_type<IDX_T>::value | arrow::is_boolean_type<IDX_T>::value>::type>
arrow::Status HashGroupBy(arrow::MemoryPool *pool,
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

//  std::map<IDX_C_T, VAL_HASHMAP_T> hash_map;
  std::unordered_map<IDX_C_T, VAL_HASHMAP_T> hash_map;
  hash_map.reserve(idx_col->length() * 0.5);

  const int chunks = idx_col->num_chunks();
  for (int chunk = 0; chunk < chunks; chunk++) {
    const int64_t len = idx_col->chunk(chunk)->length();
    const std::shared_ptr<IDX_ARRAY_T> &idx_arr = std::static_pointer_cast<IDX_ARRAY_T>
        (idx_col->chunk(chunk));
    const std::shared_ptr<VAL_ARRAY_T> &val_arr = std::static_pointer_cast<VAL_ARRAY_T>
        (val_col->chunk(chunk));

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
    using OUT_VAL_BUILDER_T = typename arrow::TypeTraits<typename VAL_KERNEL::ResultArrowType>::BuilderType;

    IDX_BUILDER_T idx_builder(pool);
    OUT_VAL_BUILDER_T val_builder(pool);
    std::shared_ptr<arrow::Array> out_idx, out_val;

    const unsigned long groups = hash_map.size();

    if (!(s = idx_builder.Reserve(groups)).ok()) {
      return s;
    }

    if (!(s = val_builder.Reserve(groups)).ok()) {
      return s;
    }

    for (auto &p:  hash_map) {
      idx_builder.UnsafeAppend(p.first);
      val_builder.UnsafeAppend(VAL_KERNEL::Finalize(&(p.second)));
    }
    hash_map.clear();

    if (!(s = idx_builder.Finish(&out_idx)).ok()) {
      return s;
    }
    if (!(s = val_builder.Finish(&out_val)).ok()) {
      return s;
    }

    output_arrays.push_back(out_idx);
    output_arrays.push_back(out_val);

  } else { // only build the value array
    using OUT_VAL_BUILDER_T = typename arrow::TypeTraits<typename VAL_KERNEL::ResultArrowType>::BuilderType;

    OUT_VAL_BUILDER_T val_builder(pool);
    std::shared_ptr<arrow::Array> out_val;

    const unsigned long groups = hash_map.size();

    if (!(s = val_builder.Reserve(groups)).ok()) {
      return s;
    }

    for (auto &p:  hash_map) {
      val_builder.UnsafeAppend(VAL_KERNEL::Finalize(&(p.second)));
    }
    hash_map.clear();

    if (!(s = val_builder.Finish(&out_val)).ok()) {
      return s;
    }

    output_arrays.push_back(out_val);
  }

  return arrow::Status::OK();
}

typedef arrow::Status
(*HashGroupByFptr)(arrow::MemoryPool *pool,
                   const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                   const std::shared_ptr<arrow::ChunkedArray> &val_col,
                   std::vector<std::shared_ptr<arrow::Array>> &output_arrays);

template<typename IDX_T, typename VAL_T, typename = typename std::enable_if<
    arrow::is_number_type<VAL_T>::value | arrow::is_boolean_type<VAL_T>::value>::type>
HashGroupByFptr ResolveOp(cylon::GroupByAggregationOp op) {
  switch (op) {
    case SUM: return &HashGroupBy<IDX_T, VAL_T, GroupByAggregationOp::SUM>;
    case COUNT: return &HashGroupBy<IDX_T, VAL_T, GroupByAggregationOp::COUNT>;
    case MIN:return &HashGroupBy<IDX_T, VAL_T, GroupByAggregationOp::MIN>;
    case MAX:return &HashGroupBy<IDX_T, VAL_T, GroupByAggregationOp::MAX>;
//    case MEAN:return &HashNaiveGroupBy<IDX_T, VAL_T, GroupByAggregationOp::MEAN>;
  }
  return nullptr;
}

template<typename IDX_ARROW_T,
    typename = typename std::enable_if<
        arrow::is_number_type<IDX_ARROW_T>::value
            | arrow::is_boolean_type<IDX_ARROW_T>::value>::type>
HashGroupByFptr PickHashGroupByFptr(const std::shared_ptr<cylon::DataType> &val_data_type,
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

/**
  * Local group by operation
  * Restrictions:
  *  - 0th col is the index col
  *  - every column has an aggregation op
  * @tparam IDX_ARROW_T index column type
  * @param table
  * @param index_col
  * @param aggregate_cols
  * @param aggregate_ops
  * @param output
  * @return
  */
template<typename IDX_ARROW_T,
    typename = typename std::enable_if<
        arrow::is_number_type<IDX_ARROW_T>::value
            | arrow::is_boolean_type<IDX_ARROW_T>::value>::type>
cylon::Status LocalHashGroupBy(const std::shared_ptr<cylon::Table> &table,
                               const std::vector<cylon::GroupByAggregationOp> &aggregate_ops,
                               std::shared_ptr<cylon::Table> &output) {
  if ((std::size_t) table->Columns() != aggregate_ops.size() + 1)
    return cylon::Status(cylon::Code::Invalid, "num cols != aggergate ops + 1");

  auto ctx = table->GetContext();
  auto a_table = table->get_table();

  arrow::Status a_status;
  arrow::MemoryPool *memory_pool = cylon::ToArrowPool(ctx);

  const int cols = a_table->num_columns();
  const std::shared_ptr<arrow::ChunkedArray> &idx_col = a_table->column(0);

  std::vector<std::shared_ptr<arrow::Array>> out_vectors;
  for (int c = 1; c < cols; c++) {
    const std::shared_ptr<arrow::ChunkedArray> &val_col = a_table->column(c);
    const std::shared_ptr<DataType> &val_data_type = table->GetColumn(c)->GetDataType();

    const HashGroupByFptr
        hash_group_by = PickHashGroupByFptr<IDX_ARROW_T>(val_data_type, aggregate_ops[c - 1]);

    if (hash_group_by != nullptr) {
      a_status = hash_group_by(memory_pool, idx_col, val_col, out_vectors);
    } else {
      return Status(Code::ExecutionError, "unable to find group by function");
    }

    if (!a_status.ok()) {
      LOG(FATAL) << "Aggregation failed!";
      return cylon::Status(static_cast<int>(a_status.code()), a_status.message());
    }
  }

  auto out_a_table = arrow::Table::Make(a_table->schema(), out_vectors);

  return cylon::Table::FromArrowTable(ctx, out_a_table, &output);
}

}

#endif //CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_OPS_HPP_

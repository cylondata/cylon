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
#include <glog/logging.h>

#include "cylon/ctx/arrow_memory_pool_utils.hpp"
#include "cylon/indexing/index.hpp"
#include "cylon/util/macros.hpp"
#include "cylon/util/arrow_utils.hpp"
#include "cylon/table.hpp"
#include "cylon/thridparty/flat_hash_map/bytell_hash_map.hpp"

namespace cylon {

/*
 * Checks if an int array is continuous. i.e. [x, x+1, x+2, ..]
 */
bool IsContinuous(const std::shared_ptr<arrow::Int64Array> &arr) {
  if (arr->length() < 2) {
    return true;
  }

  const auto *values = arr->data()->GetValues<int64_t>(1);
  for (int64_t i = 1; i < arr->length(); i++) {
    if (values[i] != values[i - 1] + 1) {
      return false;
    }
  }
  return true;
}

BaseArrowIndex::BaseArrowIndex(Table *table, int col_id, int64_t size) :
    table_(table), col_id_(col_id), size_(size), pool_(ToArrowPool(table->GetContext())) {}

BaseArrowIndex::BaseArrowIndex(int64_t size, arrow::MemoryPool *pool) :
    table_(nullptr), col_id_(kNoColumnId), size_(size), pool_(pool) {}

//std::shared_ptr<arrow::Array> BaseArrowIndex::GetIndexAsArray() {
//  return table_->get_table()->column(col_id_)->chunk(0);
//}

Status BaseArrowIndex::LocationRangeByValue(const std::shared_ptr<arrow::Scalar> &start_value,
                                            const std::shared_ptr<arrow::Scalar> &end_value,
                                            int64_t *start_index,
                                            int64_t *end_index) {
  std::shared_ptr<arrow::Int64Array> indices;
  RETURN_CYLON_STATUS_IF_FAILED(LocationByValue(start_value, &indices));
  if (!IsContinuous(indices)) {
    return {Code::KeyError, "non-unique key " + start_value->ToString()};
  }
  *start_index = indices->Value(0);

  indices.reset();
  RETURN_CYLON_STATUS_IF_FAILED(LocationByValue(end_value, &indices));
  if (!IsContinuous(indices)) {
    return {Code::KeyError, "non-unique key " + end_value->ToString()};
  }
  *end_index = indices->Value(indices->length() - 1);

  if (*start_index > *end_index) {
    return {Code::IndexError, "start index > end index"};
  }

  return Status::OK();
}

/*
 * -------------------------- Hash Index --------------------------
 */
template<typename ArrowT>
class ArrowHashIndex : public BaseArrowIndex {
  using ValueT = typename util::ArrowScalarValue<ArrowT>::ValueT;
  using MapT = typename ska::bytell_hash_map<ValueT, std::vector<int64_t>>;

 public:
//  ArrowHashIndex(std::shared_ptr<arrow::Array> index_arr, int col_id, arrow::MemoryPool *pool)
//      : BaseArrowIndex(col_id, index_arr->length(), pool), index_arr_(std::move(index_arr)) {
//    build_hash_index();
//  };

  ArrowHashIndex(Table *table, int col_id) : BaseArrowIndex(table, col_id, table->Rows()),
                                             index_arr_(table_->get_table()->column(col_id_)->chunk(0)) {
    build_hash_index();
  }

  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
                         std::shared_ptr<arrow::Int64Array> *locations) override {
    arrow::Int64Builder builder(pool_);

    // if search param is null and index_array doesn't have any nulls, return empty array
    if (!search_param->is_valid) {
      if (null_indices->empty()) {
        // no nulls in the array
        return {Code::KeyError, "Key not found"};
      } else {
        // append the null positions
        RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.AppendValues(*null_indices));
        RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.Finish(locations));
        return Status::OK();
      }
    }

    const auto &res = search_param->CastTo(index_arr_->type());
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(res.status());
    const ValueT &cast_val = util::ArrowScalarValue<ArrowT>::Extract(res.ValueOrDie());

    const auto &ret = map_->find(cast_val);
    if (ret == map_->end()) { // not found
      return {Code::KeyError, "Key not found " + search_param->ToString()};
    }

    // reserve additional space to the builder, and append values
    const std::vector<int64_t> &indices = ret->second;
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.AppendValues(indices));

    RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.Finish(locations));
    return Status::OK();
  }

  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
                         int64_t *find_index) override {
    // if search param is null and index_array doesn't have any nulls, return empty array
    if (!search_param->is_valid) {
      if (null_indices->empty()) {
        // no nulls in the array
        return {Code::KeyError, "Key not found"};
      } else {
        *find_index = (*null_indices)[0];
        return Status::OK();
      }
    }

    const auto &res = search_param->CastTo(index_arr_->type());
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(res.status());
    const ValueT &cast_val = util::ArrowScalarValue<ArrowT>::Extract(res.ValueOrDie());

    const auto &ret = map_->find(cast_val);
    if (ret != map_->end()) {
      *find_index = (ret->second)[0];
      return Status::OK();
    }
    return {Code::KeyError, "Key not found"};
  }

  Status LocationByVector(const std::shared_ptr<arrow::Array> &search_param,
                          std::shared_ptr<arrow::Int64Array> *locations) override {
    if (search_param->null_count() && null_indices->empty()) {
      // null values to be searched in a non-null index
      return {Code::KeyError, "Some keys not found"};
    }

    arrow::Int64Builder builder(pool_);
    const auto &arr_data = *search_param->data();
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(
        arrow::VisitArrayDataInline<ArrowT>(
            arr_data,
            [&](ValueT val) {
              const auto &ret = map_->find(val);
              if (ret == map_->end()) { // not found
                return arrow::Status::KeyError("key not found"); // this will break the visit loop
              }
              const std::vector<int64_t> &indices = ret->second;
              return builder.AppendValues(indices);
            },
            [&]() {  // nothing to do for nulls
              return builder.AppendValues(*null_indices);
            }));

    RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.Finish(locations));
    return Status::OK();
  }

//  Status SetIndexArray(std::shared_ptr<arrow::Array> index_arr, int col_id) override {
//    if (index_arr->length() != GetSize()) {
//      return {Code::Invalid, "new index array size != current index size"};
//    }
//    SetColId(col_id);
//    index_arr_ = std::move(index_arr);
//    build_hash_index();
//    return Status::OK();
//  }

//  const std::shared_ptr<arrow::Array> &GetIndexArray() override {
//    return table_->get_table()->column(col_id_);
//  }

  Status GetIndexAsArray(std::shared_ptr<arrow::Array> *out) override {
    *out = index_arr_;
    return Status::OK();
  }

  bool IsUnique() override {
    return unique_flag;
  }

  IndexingType GetIndexingType() override { return IndexingType::Hash; }

//  Status Slice(int64_t start, int64_t end, std::shared_ptr<BaseArrowIndex> *out_index) const override {
//    if (start >= end) {
//      return {Code::Invalid, "start >= end"};
//    }
//    auto index_arr_slice = index_arr_->Slice(start, end - start);
//    *out_index = std::make_shared<ArrowHashIndex<ArrowT>>(std::move(index_arr_slice), GetColId(), GetPool());
//    return Status::OK();
//  }

 private:
  inline Status build_hash_index() {
    map_ = std::make_shared<MapT>(index_arr_->length());

    // reserve space for nulls
    null_indices = std::make_shared<std::vector<int64_t>>();
    null_indices->reserve(index_arr_->null_count());

    int64_t idx = 0;
    const auto &arr_data = *index_arr_->data();

    arrow::VisitArrayDataInline<ArrowT>(
        arr_data,
        [&](ValueT val) {
          auto &indices = (*map_)[val];
          indices.push_back(idx);
          unique_flag |= (indices.size() > 1);
          idx++;
        },
        [&]() {  // nothing to do for nulls
          null_indices->push_back(idx);
          unique_flag |= (null_indices->size() > 1);
          idx++;
        });

    return Status::OK();
  }

  std::shared_ptr<arrow::Array> index_arr_;
  std::shared_ptr<MapT> map_ = nullptr;
  std::shared_ptr<std::vector<int64_t>> null_indices = nullptr;
  bool unique_flag = false;
};

Status BuildHashIndex(Table *table, int col_id, std::shared_ptr<BaseArrowIndex> *output) {
  if (!table->Empty() && table->get_table()->column(0)->num_chunks() > 1) {
    return {Code::Invalid, "linear index can not be built with a chunked table"};
  }

  switch (table->get_table()->field(col_id)->type()->id()) {
    case arrow::Type::BOOL:*output = std::make_shared<ArrowHashIndex<arrow::BooleanType>>(table, col_id);
      break;
    case arrow::Type::UINT8:*output = std::make_shared<ArrowHashIndex<arrow::UInt8Type>>(table, col_id);
      break;
    case arrow::Type::INT8:*output = std::make_shared<ArrowHashIndex<arrow::Int8Type>>(table, col_id);
      break;
    case arrow::Type::UINT16:*output = std::make_shared<ArrowHashIndex<arrow::UInt16Type>>(table, col_id);
      break;
    case arrow::Type::INT16:*output = std::make_shared<ArrowHashIndex<arrow::Int16Type>>(table, col_id);
      break;
    case arrow::Type::UINT32:*output = std::make_shared<ArrowHashIndex<arrow::UInt32Type>>(table, col_id);
      break;
    case arrow::Type::INT32:*output = std::make_shared<ArrowHashIndex<arrow::Int32Type>>(table, col_id);
      break;
    case arrow::Type::UINT64:*output = std::make_shared<ArrowHashIndex<arrow::UInt64Type>>(table, col_id);
      break;
    case arrow::Type::INT64:*output = std::make_shared<ArrowHashIndex<arrow::Int64Type>>(table, col_id);
      break;
    case arrow::Type::FLOAT:*output = std::make_shared<ArrowHashIndex<arrow::FloatType>>(table, col_id);
      break;
    case arrow::Type::DOUBLE:*output = std::make_shared<ArrowHashIndex<arrow::DoubleType>>(table, col_id);
      break;
    case arrow::Type::STRING:*output = std::make_shared<ArrowHashIndex<arrow::StringType>>(table, col_id);
      break;
    case arrow::Type::BINARY:*output = std::make_shared<ArrowHashIndex<arrow::BinaryType>>(table, col_id);
      break;
    case arrow::Type::LARGE_STRING:*output = std::make_shared<ArrowHashIndex<arrow::LargeStringType>>(table, col_id);
      break;
    case arrow::Type::LARGE_BINARY:*output = std::make_shared<ArrowHashIndex<arrow::LargeBinaryType>>(table, col_id);
      break;
    case arrow::Type::DATE32:*output = std::make_shared<ArrowHashIndex<arrow::Date32Type>>(table, col_id);
      break;
    case arrow::Type::DATE64:*output = std::make_shared<ArrowHashIndex<arrow::Date64Type>>(table, col_id);
      break;
    case arrow::Type::TIME32:*output = std::make_shared<ArrowHashIndex<arrow::Time32Type>>(table, col_id);
      break;
    case arrow::Type::TIME64:*output = std::make_shared<ArrowHashIndex<arrow::Time64Type>>(table, col_id);
      break;
    case arrow::Type::TIMESTAMP:*output = std::make_shared<ArrowHashIndex<arrow::TimestampType>>(table, col_id);
      break;
    case arrow::Type::INTERVAL_MONTHS:
      *output = std::make_shared<ArrowHashIndex<arrow::MonthIntervalType>>(table,
                                                                           col_id);
      break;
    case arrow::Type::INTERVAL_DAY_TIME:
      // todo DayTimeIntervalType requires a custom hasher because it's c_type is a struct
//      *output = std::make_shared<ArrowHashIndex<arrow::DayTimeIntervalType>>(table, col_id);
//      break;
    default:return {Code::NotImplemented, "Unsupported data type for hash index"};
  }

  return Status::OK();
}

/*
 * -------------------------- Range Index --------------------------
 */


Status ArrowRangeIndex::LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
                                        std::shared_ptr<arrow::Int64Array> *find_index) {
  int64_t val = 0;
  RETURN_CYLON_STATUS_IF_FAILED(LocationByValue(search_param, &val));

  const auto &res = arrow::MakeArrayFromScalar(arrow::Int64Scalar(val), 1, pool_);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(res.status());

  *find_index = std::static_pointer_cast<arrow::Int64Array>(res.ValueOrDie());
  return Status::OK();
}

Status ArrowRangeIndex::LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
                                        int64_t *find_index) {
  if (!search_param->is_valid) {
    return {Code::KeyError, "invalid search param: null"};
  }

  const auto &res = search_param->CastTo(arrow::int64());
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(res.status());
  int64_t val = util::ArrowScalarValue<arrow::Int64Type>::Extract(res.ValueOrDie());

  if (val < start_ || val >= end_) {
    return {Code::KeyError, "key not found. must be in the range [start, end)"};
  }
  *find_index = val - start_;
  return Status::OK();
}

Status ArrowRangeIndex::LocationByVector(const std::shared_ptr<arrow::Array> &search_param,
                                         std::shared_ptr<arrow::Int64Array> *filter_location) {
  if (search_param->null_count() > 0) {
    return {Code::KeyError, "invalid search param: null"};
  }

  const auto & cast_res = arrow::compute::Cast(*search_param, arrow::int64());
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(cast_res.status());
  const auto & casted_search_param = cast_res.ValueOrDie();

  const auto &res = arrow::compute::MinMax(casted_search_param);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(res.status());

  const arrow::ScalarVector &min_max = res.ValueOrDie().scalar_as<arrow::StructScalar>().value;

  int64_t min = std::static_pointer_cast<arrow::Int64Scalar>(min_max[0])->value;
  int64_t max = std::static_pointer_cast<arrow::Int64Scalar>(min_max[1])->value;

  if (min < start_ || max >= end_) {
    return {Code::KeyError, "search params are out of bounds"};
  } else {
    if (start_ != 0) {
      const auto &sub_res = arrow::compute::Subtract(casted_search_param, arrow::Datum(start_));
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(sub_res.status());
      *filter_location = sub_res.ValueOrDie().array_as<arrow::Int64Array>();
    } else {
      *filter_location = std::make_shared<arrow::Int64Array>(casted_search_param->data());
    }
    return Status::OK();
  }
}


//Status ArrowRangeIndex::SetIndexArray(std::shared_ptr<arrow::Array> index_arr, int col_id) {
//  if (index_arr->length() != GetSize()) {
//    return {Code::Invalid, "new index array size != current index size"};
//  }
//  BaseArrowIndex::SetColId(col_id);
//  index_arr_ = std::move(index_arr);
//  return Status::OK();
//}

Status ArrowRangeIndex::GetIndexAsArray(std::shared_ptr<arrow::Array> *out) {
  arrow::Int64Builder builder(pool_);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.Reserve(size_));

  for (int64_t ix = start_; ix < end_; ix += step_) {
    builder.UnsafeAppend(ix);
  }

  RETURN_CYLON_STATUS_IF_ARROW_FAILED (builder.Finish(out));
  return Status::OK();
}

Status ArrowRangeIndex::Slice(int64_t start, int64_t end_inclusive, std::shared_ptr<BaseArrowIndex> *out_index) const {
  if (start >= end_inclusive) {
    return {Code::Invalid, "start >= end"};
  }
  if (start < start_ || end_inclusive > end_) {
    return {Code::Invalid, "slice start and/or end is out of bounds"};
  }

  // + step_ to get the inclusive end
  *out_index = std::make_shared<ArrowRangeIndex>(start, end_inclusive + step_, step_, pool_);
  return Status::OK();
}

std::shared_ptr<BaseArrowIndex> BuildRangeIndex(int64_t start,
                                                int64_t end,
                                                int64_t step,
                                                arrow::MemoryPool *pool) {
  return std::make_shared<ArrowRangeIndex>(start, end, step, pool);
}

/*
 * -------------------------- Linear Index --------------------------
 */
template<typename ArrowT>
class ArrowLinearIndex : public BaseArrowIndex {
 public:
//  ArrowLinearIndex(std::shared_ptr<arrow::Array> index_array, int col_id, arrow::MemoryPool *pool)
//      : BaseArrowIndex(col_id, index_array->length(), pool), unique_flag(-1), index_array_(std::move(index_array)) {}
  static constexpr int8_t kUniqueNotCalculated = -1;

  ArrowLinearIndex(Table *table, int col_id) : BaseArrowIndex(table, col_id, table->Rows()),
                                               unique_flag(kUniqueNotCalculated),
                                               index_array_(table_->get_table()->column(col_id_)->chunk(0)) {}

  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
                         std::shared_ptr<arrow::Int64Array> *locations) override {
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(util::FindIndices<ArrowT>(index_array_, search_param, locations, pool_));
    return Status::OK();
  }

  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param, int64_t *location) override {
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(util::FindIndex<ArrowT>(index_array_, search_param, location));
    return Status::OK();
  }

  Status LocationByVector(const std::shared_ptr<arrow::Array> &search_param,
                          std::shared_ptr<arrow::Int64Array> *locations) override {
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(util::FindIndices(index_array_, search_param, locations, pool_));
    return Status::OK();
  }

//  Status SetIndexArray(std::shared_ptr<arrow::Array> index_arr, int col_id) override {
//    if (index_arr->length() != GetSize()) {
//      return {Code::Invalid, "new index array size != current index size"};
//    }
//    SetColId(col_id);
//    index_array_ = std::move(index_arr);
//    unique_flag = -1;
//    return Status::OK();
//  }

  Status GetIndexAsArray(std::shared_ptr<arrow::Array> *out) override {
    *out = index_array_;
    return Status::OK();
  }

  IndexingType GetIndexingType() override {
    return Linear;
  }

  bool IsUnique() override {
    if (unique_flag == kUniqueNotCalculated) { // not calculated
      const auto &status = util::IsUnique(index_array_, reinterpret_cast<bool *>(&unique_flag), pool_);

      if (!status.ok()) {
        unique_flag = -1; // uniqueness check failed
      }
    }

    return unique_flag == 1;
  }

//  Status Slice(int64_t start, int64_t end, std::shared_ptr<BaseArrowIndex> *out_index) const override {
//    if (start >= end) {
//      return {Code::Invalid, "start >= end"};
//    }
//    auto index_arr_slice = index_array_->Slice(start, end - start);
//    *out_index = std::make_shared<ArrowLinearIndex<ArrowT>>(std::move(index_arr_slice), GetColId(), GetPool());
//    return Status::OK();
//  }

 private:
  int8_t unique_flag; // -1 = not calculated, 0 = not unique, 1 = unique
  std::shared_ptr<arrow::Array> index_array_;
};

Status BuildLinearIndex(Table *table, int col_id, std::shared_ptr<BaseArrowIndex> *out_index) {
  if (!table->Empty() && table->get_table()->column(0)->num_chunks() > 1) {
    return {Code::Invalid, "linear index can not be built with a chunked table"};
  }

  switch (table->get_table()->field(col_id)->type()->id()) {
    case arrow::Type::BOOL:*out_index = std::make_shared<ArrowLinearIndex<arrow::BooleanType>>(table, col_id);
      break;
    case arrow::Type::UINT8:*out_index = std::make_shared<ArrowLinearIndex<arrow::UInt8Type>>(table, col_id);
      break;
    case arrow::Type::INT8:*out_index = std::make_shared<ArrowLinearIndex<arrow::Int8Type>>(table, col_id);
      break;
    case arrow::Type::UINT16:*out_index = std::make_shared<ArrowLinearIndex<arrow::UInt16Type>>(table, col_id);
      break;
    case arrow::Type::INT16:*out_index = std::make_shared<ArrowLinearIndex<arrow::Int16Type>>(table, col_id);
      break;
    case arrow::Type::UINT32:*out_index = std::make_shared<ArrowLinearIndex<arrow::UInt32Type>>(table, col_id);
      break;
    case arrow::Type::INT32:*out_index = std::make_shared<ArrowLinearIndex<arrow::Int32Type>>(table, col_id);
      break;
    case arrow::Type::UINT64:*out_index = std::make_shared<ArrowLinearIndex<arrow::UInt64Type>>(table, col_id);
      break;
    case arrow::Type::INT64:*out_index = std::make_shared<ArrowLinearIndex<arrow::Int64Type>>(table, col_id);
      break;
    case arrow::Type::FLOAT:*out_index = std::make_shared<ArrowLinearIndex<arrow::FloatType>>(table, col_id);
      break;
    case arrow::Type::DOUBLE:*out_index = std::make_shared<ArrowLinearIndex<arrow::DoubleType>>(table, col_id);
      break;
    case arrow::Type::STRING:*out_index = std::make_shared<ArrowLinearIndex<arrow::StringType>>(table, col_id);
      break;
    case arrow::Type::BINARY:*out_index = std::make_shared<ArrowLinearIndex<arrow::BinaryType>>(table, col_id);
      break;
    case arrow::Type::LARGE_STRING:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::LargeStringType>>(table, col_id);
      break;
    case arrow::Type::LARGE_BINARY:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::LargeBinaryType>>(table, col_id);
      break;
    case arrow::Type::FIXED_SIZE_BINARY:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::FixedSizeBinaryType>>(table,
                                                                                  col_id);
      break;
    case arrow::Type::DATE32:*out_index = std::make_shared<ArrowLinearIndex<arrow::Date32Type>>(table, col_id);
      break;
    case arrow::Type::DATE64:*out_index = std::make_shared<ArrowLinearIndex<arrow::Date64Type>>(table, col_id);
      break;
    case arrow::Type::TIMESTAMP:*out_index = std::make_shared<ArrowLinearIndex<arrow::TimestampType>>(table, col_id);
      break;
    case arrow::Type::TIME32:*out_index = std::make_shared<ArrowLinearIndex<arrow::Time32Type>>(table, col_id);
      break;
    case arrow::Type::TIME64:*out_index = std::make_shared<ArrowLinearIndex<arrow::Time64Type>>(table, col_id);
      break;
    case arrow::Type::INTERVAL_MONTHS:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::MonthIntervalType>>(table,
                                                                                col_id);
      break;
    case arrow::Type::INTERVAL_DAY_TIME:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::DayTimeIntervalType>>(table,
                                                                                  col_id);
      break;
    default:return {Code::NotImplemented, "Unsupported data type for linear index"};
  }

  return Status::OK();
}

Status BuildIndex(Table *table, int col_id, IndexingType indexing_type,
                  std::shared_ptr<BaseArrowIndex> *index) {
  switch (indexing_type) {
    case Range:*index = BuildRangeIndex(0, table->Rows(), 1, ToArrowPool(table->GetContext()));
      return Status::OK();
    case Linear: return BuildLinearIndex(table, col_id, index);
    case Hash: return BuildHashIndex(table, col_id, index);
    case BinaryTree:
    case BTree:
    default:return {Code::NotImplemented, "Unsupported index type"};
  }
}

//Status BuildIndex(std::shared_ptr<arrow::Array> index_array,
//                  IndexingType indexing_type,
//                  std::shared_ptr<BaseArrowIndex> *output,
//                  int col_id,
//                  arrow::MemoryPool *pool) {
//  switch (indexing_type) {
//    case Range:*output = std::make_shared<ArrowRangeIndex>(0, index_array->length(), 1, pool);
//      return Status::OK();
//    case Linear: return BuildLinearIndex(std::move(index_array), output, col_id, pool);
//    case Hash: return BuildHashIndex(std::move(index_array), output, col_id, pool);
//    case BinaryTree:
//    case BTree:
//    default:return {Code::NotImplemented, "Unsupported index type"};
//  }
//}

//Status SetIndexForTable(std::shared_ptr<Table> &table, int col_id, IndexingType indexing_type, bool drop) {
//  arrow::MemoryPool *pool = ToArrowPool(table->GetContext());
//
//  // take a copy of the arrow table
//  auto a_table = table->get_table();
//
//  std::shared_ptr<BaseArrowIndex> index;
//  RETURN_CYLON_STATUS_IF_FAILED(BuildIndex(a_table, col_id, indexing_type, &index));
//
//  // set index
//  RETURN_CYLON_STATUS_IF_FAILED(table->SetArrowIndex(std::move(index), drop));
//  return Status::OK();
//}
//
//Status SetIndexForTable(std::shared_ptr<Table> &table,
//                        std::shared_ptr<arrow::Array> array,
//                        IndexingType indexing_type,
//                        int col_id) {
//  if (array->length() != table->Rows()) {
//    return {Code::Invalid, "array length != table length"};
//  }
//
//  arrow::MemoryPool *pool = ToArrowPool(table->GetContext());
//  std::shared_ptr<BaseArrowIndex> index;
//  RETURN_CYLON_STATUS_IF_FAILED(BuildIndex(std::move(array), indexing_type, &index, col_id, pool));
//
//  RETURN_CYLON_STATUS_IF_FAILED(table->SetArrowIndex(std::move(index), /*drop_index*/false));
//  return Status::OK();
//}

} // namespace cylon








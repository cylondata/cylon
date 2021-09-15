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
  using MapT = typename std::unordered_multimap<ValueT, int64_t>;

 public:
  ArrowHashIndex(std::shared_ptr<arrow::Array> index_arr, int col_id, arrow::MemoryPool *pool)
      : BaseArrowIndex(col_id, index_arr->length(), pool), index_arr_(std::move(index_arr)) {
    build_hash_index();
  };

  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
                         std::shared_ptr<arrow::Int64Array> *locations) override {
    arrow::Int64Builder builder(GetPool());

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

    const auto &ret = map_->equal_range(cast_val);
    if (ret.first == ret.second) { // not found
      return {Code::KeyError, "Key not found"};
    }

    // reserve additional space to the builder, and append values
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.Reserve(std::distance(ret.first, ret.second)));
    for (auto it = ret.first; it != ret.second; it++) {
      builder.UnsafeAppend(it->second);
    }

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
      *find_index = ret->second;
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

    arrow::Int64Builder builder(GetPool());
    const auto &arr_data = *search_param->data();
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(
        arrow::VisitArrayDataInline<ArrowT>(
            arr_data,
            [&](ValueT val) {
              const auto &ret = map_->equal_range(val);
              if (ret.first == ret.second) { // not found
                return arrow::Status::Cancelled("key not found"); // this will break the visit loop
              }
              // reserve additional space to the builder, and append values
              RETURN_ARROW_STATUS_IF_FAILED(builder.Reserve(std::distance(ret.first, ret.second)));
              for (auto it = ret.first; it != ret.second; it++) {
                builder.UnsafeAppend(it->second);
              }
              return arrow::Status::OK();
            },
            [&]() {  // nothing to do for nulls
              return builder.AppendValues(*null_indices);
            }));

    RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.Finish(locations));
    return Status::OK();
  }

  Status SetIndexArray(std::shared_ptr<arrow::Array> index_arr, int col_id) override {
    if (index_arr->length() != GetSize()) {
      return {Code::Invalid, "new index array size != current index size"};
    }
    SetColId(col_id);
    index_arr_ = std::move(index_arr);
    build_hash_index();
    return Status::OK();
  }

  const std::shared_ptr<arrow::Array> &GetIndexArray() override {
    return index_arr_;
  }

  bool IsUnique() override {
    if (unique_flag < 0) { // not calculated
      const auto &status = util::IsUnique(index_arr_, reinterpret_cast<bool *>(&unique_flag), GetPool());

      if (!status.ok()) {
        unique_flag = -1; // uniqueness check failed
      }
    }

    return unique_flag == 1;
  }

  IndexingType GetIndexingType() override { return IndexingType::Hash; }

  Status Slice(int64_t start, int64_t end, std::shared_ptr<BaseArrowIndex> *out_index) const override {
    if (start >= end) {
      return {Code::Invalid, "start >= end"};
    }
    auto index_arr_slice = index_arr_->Slice(start, end - start);
    *out_index = std::make_shared<ArrowHashIndex<ArrowT>>(std::move(index_arr_slice), GetColId(), GetPool());
    return Status::OK();
  }

 private:
  Status build_hash_index() {
    map_ = std::make_shared<MapT>(index_arr_->length());

    // reserve space for nulls
    null_indices = std::make_shared<std::vector<int64_t>>();
    null_indices->reserve(index_arr_->null_count());

    int64_t idx = 0;
    const auto &arr_data = *index_arr_->data();

    arrow::VisitArrayDataInline<ArrowT>(
        arr_data,
        [&](ValueT val) {
          map_->template emplace(val, idx);
          idx++;
        },
        [&]() {  // nothing to do for nulls
          null_indices->push_back(idx);
          idx++;
        });

    return Status::OK();
  }

  std::shared_ptr<arrow::Array> index_arr_;
  std::shared_ptr<MapT> map_ = nullptr;
  std::shared_ptr<std::vector<int64_t>> null_indices = nullptr;
  int8_t unique_flag = -1;
};

Status BuildHashIndex(const std::shared_ptr<arrow::Table> &table, int col_id,
                      std::shared_ptr<BaseArrowIndex> *output, arrow::MemoryPool *pool) {
  std::shared_ptr<arrow::Array> idx_arr;
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(util::GetConcatenatedColumn(table, col_id, &idx_arr, pool));

  switch (table->column(col_id)->type()->id()) {
    case arrow::Type::BOOL:
      *output = std::make_shared<ArrowHashIndex<arrow::BooleanType>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::UINT8:
      *output = std::make_shared<ArrowHashIndex<arrow::UInt8Type>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::INT8:
      *output = std::make_shared<ArrowHashIndex<arrow::Int8Type>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::UINT16:
      *output = std::make_shared<ArrowHashIndex<arrow::UInt16Type>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::INT16:
      *output = std::make_shared<ArrowHashIndex<arrow::Int16Type>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::UINT32:
      *output = std::make_shared<ArrowHashIndex<arrow::UInt16Type>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::INT32:
      *output = std::make_shared<ArrowHashIndex<arrow::Int32Type>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::UINT64:
      *output = std::make_shared<ArrowHashIndex<arrow::UInt16Type>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::INT64:
      *output = std::make_shared<ArrowHashIndex<arrow::Int64Type>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::FLOAT:
      *output = std::make_shared<ArrowHashIndex<arrow::FloatType>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::DOUBLE:
      *output = std::make_shared<ArrowHashIndex<arrow::DoubleType>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::STRING:
      *output = std::make_shared<ArrowHashIndex<arrow::StringType>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::BINARY:
      *output = std::make_shared<ArrowHashIndex<arrow::BinaryType>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::DATE32:
      *output = std::make_shared<ArrowHashIndex<arrow::Date32Type>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::DATE64:
      *output = std::make_shared<ArrowHashIndex<arrow::Date64Type>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::TIME32:
      *output = std::make_shared<ArrowHashIndex<arrow::Time32Type>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::TIME64:
      *output = std::make_shared<ArrowHashIndex<arrow::Time64Type>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::TIMESTAMP:
      *output = std::make_shared<ArrowHashIndex<arrow::TimestampType>>(std::move(idx_arr),
                                                                       col_id,
                                                                       pool);
      break;
    case arrow::Type::INTERVAL_MONTHS:
      *output = std::make_shared<ArrowHashIndex<arrow::MonthIntervalType>>(std::move(idx_arr),
                                                                           col_id,
                                                                           pool);
      break;
    case arrow::Type::INTERVAL_DAY_TIME:
      // todo DayTimeIntervalType requires a custom hasher because it's c_type is a struct
//      *output = std::make_shared<ArrowHashIndex<arrow::DayTimeIntervalType>>(std::move(idx_arr), col_id, pool);
//      break;
    default:return {Code::NotImplemented, "Unsupported data type for hash index"};
  }

  return Status::OK();
}

/*
 * -------------------------- Range Index --------------------------
 */
class ArrowRangeIndex : public BaseArrowIndex {
 public:
  ArrowRangeIndex(int64_t start, int64_t end, int64_t step, arrow::MemoryPool *pool)
      : BaseArrowIndex(kNoColumnId, (end - start) / step, pool), start_(start), end_(end), step_(step) {}

  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
                         std::shared_ptr<arrow::Int64Array> *find_index) override {
    if (search_param->type->id() != arrow::Type::INT64) {
      return {Code::Invalid, "RangeIndex values should be Int64 type, given " + search_param->type->ToString()};
    }

    auto casted_search_param = std::static_pointer_cast<arrow::Int64Scalar>(search_param);
    int64_t val = casted_search_param->value;

    if (val < start_ || val >= end_) {
      return {Code::KeyError, "Invalid Key, it must be in the range [start, end)"};
    }

    const auto &res = arrow::MakeArrayFromScalar(*search_param, 1, GetPool());
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(res.status());
    *find_index = std::static_pointer_cast<arrow::Int64Array>(res.ValueOrDie());

    return Status::OK();
  }

  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
                         int64_t *find_index) override {
    if (search_param->type->id() != arrow::Type::INT64) {
      return {Code::Invalid, "RangeIndex values should be Int64 type, given " + search_param->type->ToString()};
    }
    auto casted_search_param = std::static_pointer_cast<arrow::Int64Scalar>(search_param);
    int64_t val = casted_search_param->value;
    if (val < start_ || val >= end_) {
      return {Code::KeyError, "Invalid Key, it must be in the range of 0, num of records"};
    }
    *find_index = val;
    return Status::OK();
  }

  Status LocationByVector(const std::shared_ptr<arrow::Array> &search_param,
                          std::shared_ptr<arrow::Int64Array> *filter_location) override {
    if (search_param->type()->id() != arrow::Type::INT64) {
      return {Code::Invalid, "RangeIndex values should be Int64 type, given " + search_param->type()->ToString()};
    }

    const auto &res = arrow::compute::MinMax(search_param);
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(res.status());

    const arrow::ScalarVector &min_max = res.ValueOrDie().scalar_as<arrow::StructScalar>().value;

    int64_t min = std::static_pointer_cast<arrow::Int64Scalar>(min_max[0])->value;
    int64_t max = std::static_pointer_cast<arrow::Int64Scalar>(min_max[1])->value;

    if (min < start_ || max >= end_) {
      return {Code::Invalid, "search params are out of bounds"};
    } else {
      *filter_location = std::static_pointer_cast<arrow::Int64Array>(search_param);
      return Status::OK();
    }
  }

  Status SetIndexArray(std::shared_ptr<arrow::Array> index_arr, int col_id) override {
    if (index_arr->length() != GetSize()) {
      return {Code::Invalid, "new index array size != current index size"};
    }
    BaseArrowIndex::SetColId(col_id);
    index_arr_ = std::move(index_arr);
    return Status::OK();
  }

  const std::shared_ptr<arrow::Array> &GetIndexArray() override {
    // if the index array is already populated, return it!
    if (index_arr_) {
      return index_arr_;
    }

    arrow::Int64Builder builder(GetPool());
    if (!builder.Reserve(GetSize()).ok()) {
      return index_arr_; // if reservation fails, return nullptr
    }

    for (int64_t ix = start_; ix < end_; ix += step_) {
      builder.UnsafeAppend(ix);
    }

    if (!builder.Finish(&index_arr_).ok()) {
      index_arr_ = nullptr; // if builder finishing fails, return nullptr
    }

    return index_arr_;
  }

  IndexingType GetIndexingType() override {
    return Range;
  }

  bool IsUnique() override {
    return true;
  }

  Status Slice(int64_t start, int64_t end, std::shared_ptr<BaseArrowIndex> *out_index) const override {
    if (start >= end) {
      return {Code::Invalid, "start >= end"};
    }
    *out_index = std::make_shared<ArrowRangeIndex>(start, end, step_, GetPool());

    // if the index_arr_ is already populated for index, do a zero-copy slice
    if (index_arr_) {
      (*out_index)->SetIndexArray(index_arr_->Slice(start, end - start), kNoColumnId);
    }
    return Status::OK();
  }

  int64_t GetStart() const { return start_; }
  int64_t GetEnd() const { return end_; }
  int64_t GetStep() const { return step_; }

 private:
  int64_t start_ = 0;
  int64_t end_ = 0;
  int64_t step_ = 1;
  std::shared_ptr<arrow::Array> index_arr_ = nullptr;
};

std::shared_ptr<BaseArrowIndex> BuildRangeIndex(const std::shared_ptr<arrow::Table> &input_table,
                                                int64_t start,
                                                int64_t end,
                                                int64_t step,
                                                arrow::MemoryPool *pool) {
  return std::make_shared<ArrowRangeIndex>(start, end < 0 ? input_table->num_rows() : end, step, pool);
}

/*
 * -------------------------- Linear Index --------------------------
 */
template<typename ArrowT>
class ArrowLinearIndex : public BaseArrowIndex {
 public:
  ArrowLinearIndex(std::shared_ptr<arrow::Array> index_array, int col_id, arrow::MemoryPool *pool)
      : BaseArrowIndex(col_id, index_array->length(), pool), unique_flag(-1), index_array_(std::move(index_array)) {}

  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
                         std::shared_ptr<arrow::Int64Array> *locations) override {
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(util::FindIndices<ArrowT>(index_array_, search_param, locations, GetPool()));
    return Status::OK();
  }

  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param, int64_t *location) override {
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(util::FindIndex<ArrowT>(index_array_, search_param, location));
    return Status::OK();
  }

  Status LocationByVector(const std::shared_ptr<arrow::Array> &search_param,
                          std::shared_ptr<arrow::Int64Array> *locations) override {
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(util::FindIndices(index_array_, search_param, locations, GetPool()));
    return Status::OK();
  }

  Status SetIndexArray(std::shared_ptr<arrow::Array> index_arr, int col_id) override {
    if (index_arr->length() != GetSize()) {
      return {Code::Invalid, "new index array size != current index size"};
    }
    SetColId(col_id);
    index_array_ = std::move(index_arr);
    unique_flag = -1;
    return Status::OK();
  }

  const std::shared_ptr<arrow::Array> &GetIndexArray() override {
    return index_array_;
  }

  IndexingType GetIndexingType() override {
    return Linear;
  }

  bool IsUnique() override {
    if (unique_flag < 0) { // not calculated
      const auto &status = util::IsUnique(index_array_, reinterpret_cast<bool *>(&unique_flag), GetPool());

      if (!status.ok()) {
        unique_flag = -1; // uniqueness check failed
      }
    }

    return unique_flag == 1;
  }

  Status Slice(int64_t start, int64_t end, std::shared_ptr<BaseArrowIndex> *out_index) const override {
    if (start >= end) {
      return {Code::Invalid, "start >= end"};
    }
    auto index_arr_slice = index_array_->Slice(start, end - start);
    *out_index = std::make_shared<ArrowLinearIndex<ArrowT>>(std::move(index_arr_slice), GetColId(), GetPool());
    return Status::OK();
  }

 private:
  int8_t unique_flag; // -1 = not calculated, 0 = not unique, 1 = unique
  std::shared_ptr<arrow::Array> index_array_;
};

Status BuildLinearIndex(const std::shared_ptr<arrow::Table> &table, int col_id,
                        std::shared_ptr<BaseArrowIndex> *out_index, arrow::MemoryPool *pool) {
  std::shared_ptr<arrow::Array> idx_arr;
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(util::GetConcatenatedColumn(table, col_id, &idx_arr, pool));

  return BuildLinearIndex(std::move(idx_arr), out_index, col_id, pool);
}

Status BuildLinearIndex(std::shared_ptr<arrow::Array> idx_arr, std::shared_ptr<BaseArrowIndex> *out_index,
                        int col_id, arrow::MemoryPool *pool) {
  switch (idx_arr->type()->id()) {
    case arrow::Type::BOOL:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::BooleanType>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::UINT8:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::UInt8Type>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::INT8:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::Int8Type>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::UINT16:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::UInt16Type>>(std::move(idx_arr),
                                                                         col_id,
                                                                         pool);
      break;
    case arrow::Type::INT16:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::Int16Type>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::UINT32:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::UInt32Type>>(std::move(idx_arr),
                                                                         col_id,
                                                                         pool);
      break;
    case arrow::Type::INT32:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::Int32Type>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::UINT64:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::UInt64Type>>(std::move(idx_arr),
                                                                         col_id,
                                                                         pool);
      break;
    case arrow::Type::INT64:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::Int64Type>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::FLOAT:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::FloatType>>(std::move(idx_arr), col_id, pool);
      break;
    case arrow::Type::DOUBLE:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::DoubleType>>(std::move(idx_arr),
                                                                         col_id,
                                                                         pool);
      break;
    case arrow::Type::STRING:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::StringType>>(std::move(idx_arr),
                                                                         col_id,
                                                                         pool);
      break;
    case arrow::Type::BINARY:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::BinaryType>>(std::move(idx_arr),
                                                                         col_id,
                                                                         pool);
      break;
    case arrow::Type::LARGE_STRING:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::LargeStringType>>(std::move(idx_arr),
                                                                              col_id,
                                                                              pool);
      break;
    case arrow::Type::LARGE_BINARY:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::LargeBinaryType>>(std::move(idx_arr),
                                                                              col_id,
                                                                              pool);
      break;
    case arrow::Type::FIXED_SIZE_BINARY:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::FixedSizeBinaryType>>(std::move(idx_arr),
                                                                                  col_id,
                                                                                  pool);
      break;
    case arrow::Type::DATE32:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::Date32Type>>(std::move(idx_arr),
                                                                         col_id,
                                                                         pool);
      break;
    case arrow::Type::DATE64:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::Date64Type>>(std::move(idx_arr),
                                                                         col_id,
                                                                         pool);
      break;
    case arrow::Type::TIMESTAMP:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::TimestampType>>(std::move(idx_arr),
                                                                            col_id,
                                                                            pool);
      break;
    case arrow::Type::TIME32:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::Time32Type>>(std::move(idx_arr),
                                                                         col_id,
                                                                         pool);
      break;
    case arrow::Type::TIME64:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::Time64Type>>(std::move(idx_arr),
                                                                         col_id,
                                                                         pool);
      break;
    case arrow::Type::INTERVAL_MONTHS:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::MonthIntervalType>>(std::move(idx_arr),
                                                                                col_id,
                                                                                pool);
      break;
    case arrow::Type::INTERVAL_DAY_TIME:
      *out_index = std::make_shared<ArrowLinearIndex<arrow::DayTimeIntervalType>>(std::move(idx_arr), col_id, pool);
      break;
    default:return {Code::NotImplemented, "Unsupported data type for linear index"};
  }

  return Status::OK();
}

Status BuildIndex(const std::shared_ptr<arrow::Table> &table,
                  int col_id,
                  IndexingType indexing_type,
                  std::shared_ptr<BaseArrowIndex> *index,
                  arrow::MemoryPool *pool) {
  switch (indexing_type) {
    case Range:*index = BuildRangeIndex(table, 0, -1, 1, pool);
      return Status::OK();
    case Linear: return BuildLinearIndex(table, col_id, index);
    case Hash: return BuildHashIndex(table, col_id, index);
    case BinaryTree:
    case BTree:
    default:return {Code::NotImplemented, "Unsupported index type"};
  }
}

Status SetIndexForTable(IndexingType indexing_type, const std::shared_ptr<Table> &table,
                        int col_id, std::shared_ptr<Table> *out_table, bool drop) {
  arrow::MemoryPool *pool = ToArrowPool(table->GetContext());

  // take a copy of the arrow table
  auto a_table = table->get_table();

  std::shared_ptr<BaseArrowIndex> index;
  RETURN_CYLON_STATUS_IF_FAILED(BuildIndex(a_table, col_id, indexing_type, &index, pool));

  // create cylon table
  *out_table = std::make_shared<Table>(table->GetContext(), std::move(a_table));
  RETURN_CYLON_STATUS_IF_FAILED((*out_table)->SetArrowIndex(std::move(index), drop));
  return Status::OK();
}

} // namespace cylon








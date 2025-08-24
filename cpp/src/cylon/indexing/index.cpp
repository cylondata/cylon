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
#include <cylon/indexing/index.hpp>
#include <cylon/table.hpp>

namespace cylon {

std::unique_ptr<ArrowIndexKernel> CreateArrowHashIndexKernel(std::shared_ptr<arrow::Table> input_table,
															 int index_column) {
  switch (input_table->column(index_column)->type()->id()) {
    case arrow::Type::NA:
      return nullptr;
    case arrow::Type::BOOL:
      return std::make_unique<BoolArrowHashIndexKernel>();
    case arrow::Type::UINT8:
      return std::make_unique<UInt8ArrowHashIndexKernel>();
    case arrow::Type::INT8:
      return std::make_unique<Int8ArrowHashIndexKernel>();
    case arrow::Type::UINT16:
      return std::make_unique<UInt16ArrowHashIndexKernel>();
    case arrow::Type::INT16:
      return std::make_unique<Int16ArrowHashIndexKernel>();
    case arrow::Type::UINT32:
      return std::make_unique<UInt32ArrowHashIndexKernel>();
    case arrow::Type::INT32:
      return std::make_unique<Int32ArrowHashIndexKernel>();
    case arrow::Type::UINT64:
      return std::make_unique<UInt64ArrowHashIndexKernel>();
    case arrow::Type::INT64:
      return std::make_unique<Int64ArrowHashIndexKernel>();
    case arrow::Type::HALF_FLOAT:
      return std::make_unique<HalfFloatArrowHashIndexKernel>();
    case arrow::Type::FLOAT:
      return std::make_unique<FloatArrowHashIndexKernel>();
    case arrow::Type::DOUBLE:
      return std::make_unique<DoubleArrowHashIndexKernel>();
    case arrow::Type::STRING:
      return std::make_unique<StringArrowHashIndexKernel>();
    case arrow::Type::BINARY:
      return std::make_unique<BinaryArrowHashIndexKernel>();
    case arrow::Type::DATE32:
      return std::make_unique<Date32ArrowHashIndexKernel>();
    case arrow::Type::DATE64:
      return std::make_unique<Date64ArrowHashIndexKernel>();
    case arrow::Type::TIME32:
      return std::make_unique<Time32ArrowHashIndexKernel>();
    case arrow::Type::TIME64:
      return std::make_unique<Time64ArrowHashIndexKernel>();
    case arrow::Type::TIMESTAMP:
      return std::make_unique<TimestampArrowHashIndexKernel>();
    default:
      return std::make_unique<ArrowRangeIndexKernel>();
  }
}

std::unique_ptr<ArrowIndexKernel> CreateArrowIndexKernel(std::shared_ptr<arrow::Table> input_table, int index_column) {
  // TODO: fix the criterion to check the kernel creation method
  if (index_column == -1) {
    return std::make_unique<ArrowRangeIndexKernel>();
  } else {
    return CreateArrowHashIndexKernel(input_table, index_column);
  }
}

bool CompareArraysForUniqueness(const std::shared_ptr<arrow::Array> &index_arr) {
  auto result = arrow::compute::Unique(index_arr);
  if (!result.status().ok()) {
    LOG(ERROR) << "Error occurred in index array unique check";
    return false;
  }
  auto unique_arr = result.ValueOrDie();
  return unique_arr->length() == index_arr->length();
}

LinearArrowIndexKernel::LinearArrowIndexKernel() {}

Status LinearArrowIndexKernel::BuildIndex(arrow::MemoryPool *pool,
										  std::shared_ptr<arrow::Table> &input_table,
										  const int index_column,
										  std::shared_ptr<BaseArrowIndex> &base_arrow_index) {
  std::shared_ptr<arrow::Array> index_array;

  if (input_table->column(0)->num_chunks() > 1) {
    const arrow::Result<std::shared_ptr<arrow::Table>> &res = input_table->CombineChunks(pool);
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(res.status());
    input_table = res.ValueOrDie();
  }

  index_array = cylon::util::GetChunkOrEmptyArray(input_table->column(index_column), 0);
  base_arrow_index = std::make_shared<ArrowLinearIndex>(index_column, input_table->num_rows(), pool, index_array);

  return cylon::Status::OK();
}

ArrowLinearIndex::ArrowLinearIndex(int col_id, int size, std::shared_ptr<CylonContext> &ctx) : BaseArrowIndex(col_id,
																											  size,
																											  ctx) {}

ArrowLinearIndex::ArrowLinearIndex(int col_id, int size, arrow::MemoryPool *pool) : BaseArrowIndex(col_id,
																								   size,
																								   pool) {}

Status ArrowLinearIndex::LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
										 std::vector<int64_t> &find_index) {

  auto cast_val = search_param->CastTo(index_array_->type()).ValueOrDie();
  for (int64_t ix = 0; ix < index_array_->length(); ix++) {
    auto val = index_array_->GetScalar(ix).ValueOrDie();
    if (cast_val->Equals(*val)) {
      find_index.push_back(ix);
    }
  }
  return Status::OK();
}
Status ArrowLinearIndex::LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param, int64_t *find_index) {
  auto cast_val = search_param->CastTo(index_array_->type()).ValueOrDie();
  for (int64_t ix = 0; ix < index_array_->length(); ix++) {
    auto val = index_array_->GetScalar(ix).ValueOrDie();
    if (cast_val->Equals(*val)) {
      *find_index = ix;
      break;
    }
  }
  return Status::OK();
}
Status ArrowLinearIndex::LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
										 const std::shared_ptr<arrow::Table> &input,
										 std::vector<int64_t> &filter_location,
										 std::shared_ptr<arrow::Table> &output) {
  arrow::Status arrow_status;
  cylon::Status status;
  std::shared_ptr<arrow::Array> out_idx;
  arrow::compute::ExecContext fn_ctx(GetPool());
  arrow::Int64Builder idx_builder(GetPool());
  RETURN_CYLON_STATUS_IF_FAILED(LocationByValue(search_param, filter_location));
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(idx_builder.AppendValues(filter_location));
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(idx_builder.Finish(&out_idx));
  arrow::Result<arrow::Datum>
	  result = arrow::compute::Take(input, out_idx, arrow::compute::TakeOptions::Defaults(), &fn_ctx);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
  output = result.ValueOrDie().table();
  return Status::OK();
}

std::shared_ptr<arrow::Array> ArrowLinearIndex::GetIndexAsArray() {
  return index_array_;
}

void ArrowLinearIndex::SetIndexArray(const std::shared_ptr<arrow::Array> &index_arr) {
  index_array_ = index_arr;
}

std::shared_ptr<arrow::Array> ArrowLinearIndex::GetIndexArray() {
  return index_array_;
}

int ArrowLinearIndex::GetColId() const {
  return BaseArrowIndex::GetColId();
}

int ArrowLinearIndex::GetSize() const {
  return index_array_->length();
}

IndexingType ArrowLinearIndex::GetIndexingType() {
  return Linear;
}

arrow::MemoryPool *ArrowLinearIndex::GetPool() const {
  return BaseArrowIndex::GetPool();
}

bool ArrowLinearIndex::IsUnique() {
  const bool is_unique = CompareArraysForUniqueness(index_array_);
  return is_unique;
}

Status ArrowLinearIndex::LocationByVector(const std::shared_ptr<arrow::Array> &search_param,
										  std::vector<int64_t> &filter_location) {

  auto cast_search_param_result = arrow::compute::Cast(search_param, index_array_->type());
  auto const search_param_array = cast_search_param_result.ValueOrDie().make_array();
  auto const res_isin_filter = arrow::compute::IsIn(index_array_, search_param_array);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(res_isin_filter.status());
  auto res_isin_filter_val = res_isin_filter.ValueOrDie();
  std::shared_ptr<arrow::Array> arr_isin = res_isin_filter_val.make_array();
  std::shared_ptr<arrow::BooleanArray> arr_isin_bool_array = std::static_pointer_cast<arrow::BooleanArray>(arr_isin);
  for (int64_t ix = 0; ix < arr_isin_bool_array->length(); ix++) {
    auto val = arr_isin_bool_array->Value(ix);
    if (val) {
      filter_location.push_back(ix);
    }
  }
  return Status::OK();
}

int BaseArrowIndex::GetColId() const {
  return col_id_;
}

int BaseArrowIndex::GetSize() const {
  return size_;
}

arrow::MemoryPool *BaseArrowIndex::GetPool() const {
  return pool_;
}

Status ArrowRangeIndex::LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
										std::vector<int64_t> &find_index) {
  std::shared_ptr<arrow::Int64Scalar> casted_search_param = std::static_pointer_cast<arrow::Int64Scalar>(search_param);
  int64_t val = casted_search_param->value;
  if (!(val >= start_ && val < end_)) {
    return Status(cylon::Code::KeyError, "Invalid Key, it must be in the range of 0, num of records");
  }
  find_index.push_back(val);
  return Status::OK();
}

Status ArrowRangeIndex::LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param, int64_t *find_index) {
  std::shared_ptr<arrow::Int64Scalar> casted_search_param = std::static_pointer_cast<arrow::Int64Scalar>(search_param);
  int64_t val = casted_search_param->value;
  if (!(val >= start_ && val < end_)) {
    return Status(cylon::Code::KeyError, "Invalid Key, it must be in the range of 0, num of records");
  }
  *find_index = val;
  return Status::OK();
}

Status ArrowRangeIndex::LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
										const std::shared_ptr<arrow::Table> &input,
										std::vector<int64_t> &filter_location,
										std::shared_ptr<arrow::Table> &output) {
  arrow::Status arrow_status;
  cylon::Status status;
  std::shared_ptr<arrow::Array> out_idx;
  arrow::compute::ExecContext fn_ctx(GetPool());
  arrow::Int64Builder idx_builder(GetPool());
  RETURN_CYLON_STATUS_IF_FAILED(LocationByValue(search_param, filter_location));
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(idx_builder.AppendValues(filter_location));
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(idx_builder.Finish(&out_idx));
  arrow::Result<arrow::Datum>
	  result = arrow::compute::Take(input, out_idx, arrow::compute::TakeOptions::Defaults(), &fn_ctx);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
  output = result.ValueOrDie().table();
  return Status::OK();
}

std::shared_ptr<arrow::Array> ArrowRangeIndex::GetIndexAsArray() {
  arrow::Status arrow_status;
  auto pool = GetPool();
  arrow::Int64Builder builder(pool);
  std::vector<int64_t> vec(GetSize());
  for (int64_t ix = 0; ix < GetSize(); ix += GetStep()) {
    vec[ix] = ix;
  }
  arrow_status = builder.AppendValues(vec);
  if (!arrow_status.ok()) {
    LOG(ERROR) << "Error occurred when generating range index values with array builder";
  }
  arrow_status = builder.Finish(&index_arr_);
  if (!arrow_status.ok()) {
    LOG(ERROR) << "Error occurred in retrieving index";
    return nullptr;
  }
  return index_arr_;
}

void ArrowRangeIndex::SetIndexArray(const std::shared_ptr<arrow::Array> &index_arr) {
  index_arr_ = index_arr;
}

std::shared_ptr<arrow::Array> ArrowRangeIndex::GetIndexArray() {
  return GetIndexAsArray();
}

int ArrowRangeIndex::GetColId() const {
  return BaseArrowIndex::GetColId();
}

int ArrowRangeIndex::GetSize() const {
  return end_;
}

IndexingType ArrowRangeIndex::GetIndexingType() {
  return Range;
}

arrow::MemoryPool *ArrowRangeIndex::GetPool() const {
  return BaseArrowIndex::GetPool();
}

bool ArrowRangeIndex::IsUnique() {
  return true;
}

ArrowRangeIndex::ArrowRangeIndex(int start, int size, int step, arrow::MemoryPool *pool) : BaseArrowIndex(0,
																										  size,
																										  pool),
																						   start_(start),
																						   end_(size),
																						   step_(step) {

}

int ArrowRangeIndex::GetStart() const {
  return start_;
}

int ArrowRangeIndex::GetAnEnd() const {
  return end_;
}

int ArrowRangeIndex::GetStep() const {
  return step_;
}

Status ArrowRangeIndex::LocationByVector(const std::shared_ptr<arrow::Array> &search_param,
										 std::vector<int64_t> &filter_location) {
  std::shared_ptr<arrow::Int64Array> cast_search_param = std::static_pointer_cast<arrow::Int64Array>(search_param);
  for (int64_t ix = 0; ix < cast_search_param->length(); ix++) {
    int64_t index_value = cast_search_param->Value(ix);
    filter_location.push_back(index_value);
  }
  return Status::OK();
};

ArrowRangeIndexKernel::ArrowRangeIndexKernel() {

}

Status ArrowRangeIndexKernel::BuildIndex(arrow::MemoryPool *pool,
										 std::shared_ptr<arrow::Table> &input_table,
										 const int index_column,
										 std::shared_ptr<BaseArrowIndex> &base_arrow_index) {
  CYLON_UNUSED(index_column);
                       
  base_arrow_index = std::make_shared<ArrowRangeIndex>(0, input_table->num_rows(), 1, pool);
  return cylon::Status::OK();
}
}








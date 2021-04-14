#include "index.hpp"
#include "table.hpp"

namespace cylon {

//std::unique_ptr<IndexKernel> CreateHashIndexKernel(std::shared_ptr<arrow::Table> input_table, int index_column) {
//  switch (input_table->column(index_column)->type()->id()) {
//
//	case arrow::Type::NA:return nullptr;
//	case arrow::Type::BOOL:return std::make_unique<BoolHashIndexKernel>();
//	case arrow::Type::UINT8:return std::make_unique<UInt8HashIndexKernel>();
//	case arrow::Type::INT8:return std::make_unique<Int8HashIndexKernel>();
//	case arrow::Type::UINT16:return std::make_unique<UInt16HashIndexKernel>();
//	case arrow::Type::INT16:return std::make_unique<Int16HashIndexKernel>();
//	case arrow::Type::UINT32:return std::make_unique<UInt32HashIndexKernel>();
//	case arrow::Type::INT32:return std::make_unique<Int32HashIndexKernel>();
//	case arrow::Type::UINT64:return std::make_unique<UInt64HashIndexKernel>();
//	case arrow::Type::INT64: return std::make_unique<Int64HashIndexKernel>();
//	case arrow::Type::HALF_FLOAT:return std::make_unique<HalfFloatHashIndexKernel>();
//	case arrow::Type::FLOAT:return std::make_unique<FloatHashIndexKernel>();
//	case arrow::Type::DOUBLE:return std::make_unique<DoubleHashIndexKernel>();
//	case arrow::Type::STRING:return std::make_unique<StringHashIndexKernel>();
//	case arrow::Type::BINARY:return nullptr;//std::make_unique<BinaryHashIndexKernel>();
//	default: return std::make_unique<GenericRangeIndexKernel>();
//  }
//
//}

std::unique_ptr<ArrowIndexKernel> CreateArrowHashIndexKernel(std::shared_ptr<arrow::Table> input_table,
															 int index_column) {
  switch (input_table->column(index_column)->type()->id()) {

	case arrow::Type::NA:return nullptr;
	case arrow::Type::BOOL:return std::make_unique<BoolArrowHashIndexKernel>();
	case arrow::Type::UINT8:return std::make_unique<UInt8ArrowHashIndexKernel>();
	case arrow::Type::INT8:return std::make_unique<Int8ArrowHashIndexKernel>();
	case arrow::Type::UINT16:return std::make_unique<UInt16ArrowHashIndexKernel>();
	case arrow::Type::INT16:return std::make_unique<Int16ArrowHashIndexKernel>();
	case arrow::Type::UINT32:return std::make_unique<UInt32ArrowHashIndexKernel>();
	case arrow::Type::INT32:return std::make_unique<Int32ArrowHashIndexKernel>();
	case arrow::Type::UINT64:return std::make_unique<UInt64ArrowHashIndexKernel>();
	case arrow::Type::INT64: return std::make_unique<Int64ArrowHashIndexKernel>();
	case arrow::Type::HALF_FLOAT:return std::make_unique<HalfFloatArrowHashIndexKernel>();
	case arrow::Type::FLOAT:return std::make_unique<FloatArrowHashIndexKernel>();
	case arrow::Type::DOUBLE:return std::make_unique<DoubleArrowHashIndexKernel>();
	case arrow::Type::STRING:return std::make_unique<StringArrowHashIndexKernel>();
	case arrow::Type::BINARY:return nullptr;//std::make_unique<BinaryHashIndexKernel>();
	default: return std::make_unique<ArrowRangeIndexKernel>();
  }

}

//std::unique_ptr<IndexKernel> CreateLinearIndexKernel(std::shared_ptr<arrow::Table> input_table, int index_column) {
//  switch (input_table->column(index_column)->type()->id()) {
//
//	case arrow::Type::NA:return nullptr;
//	case arrow::Type::BOOL:return std::make_unique<BoolLinearIndexKernel>();
//	case arrow::Type::UINT8:return std::make_unique<UInt8LinearIndexKernel>();
//	case arrow::Type::INT8:return std::make_unique<Int8LinearIndexKernel>();
//	case arrow::Type::UINT16:return std::make_unique<UInt16LinearIndexKernel>();
//	case arrow::Type::INT16:return std::make_unique<Int16LinearIndexKernel>();
//	case arrow::Type::UINT32:return std::make_unique<UInt32LinearIndexKernel>();
//	case arrow::Type::INT32:return std::make_unique<Int32LinearIndexKernel>();
//	case arrow::Type::UINT64:return std::make_unique<UInt64LinearIndexKernel>();
//	case arrow::Type::INT64: return std::make_unique<Int64LinearIndexKernel>();
//	case arrow::Type::HALF_FLOAT:return std::make_unique<HalfFloatLinearIndexKernel>();
//	case arrow::Type::FLOAT:return std::make_unique<FloatLinearIndexKernel>();
//	case arrow::Type::DOUBLE:return std::make_unique<DoubleLinearIndexKernel>();
//	case arrow::Type::STRING:return std::make_unique<StringLinearIndexKernel>();
//	case arrow::Type::BINARY:return nullptr;//std::make_unique<BinaryLinearIndexKernel>();
//	default: return std::make_unique<GenericRangeIndexKernel>();
//  }
//
//}

//std::unique_ptr<IndexKernel> CreateIndexKernel(std::shared_ptr<arrow::Table> input_table, int index_column) {
//  // TODO: fix the criterion to check the kernel creation method
//  if (index_column == -1) {
//	return std::make_unique<GenericRangeIndexKernel>();
//  } else {
//	return CreateHashIndexKernel(input_table, index_column);
//  }
//}

std::unique_ptr<ArrowIndexKernel> CreateArrowIndexKernel(std::shared_ptr<arrow::Table> input_table, int index_column) {
  // TODO: fix the criterion to check the kernel creation method
  if (index_column == -1) {
	return std::make_unique<ArrowRangeIndexKernel>();
  } else {
	return CreateArrowHashIndexKernel(input_table, index_column);
  }
}

cylon::Status CompareArraysForUniqueness(std::shared_ptr<arrow::Array> &index_arr, bool &is_unique) {
  Status s;
  auto result = arrow::compute::Unique(index_arr);

  if (!result.ok()) {
	LOG(ERROR) << "Error occurred in unique operation on index array";
	RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
  }
  auto unique_arr = result.ValueOrDie();

  is_unique = unique_arr->length() == index_arr->length();

  return cylon::Status::OK();
}

//void BuildRangeIndexArray(int start,
//						  int end,
//						  int step,
//						  arrow::MemoryPool *pool,
//						  std::shared_ptr<arrow::Array> &index_arr) {
//  arrow::Status ar_status;
//  arrow::Int64Builder builder(pool);
//  int64_t capacity = int64_t((end - start) / step);
//  if (!(ar_status = builder.Reserve(capacity)).ok()) {
//	LOG(ERROR) << "Error occurred in reserving memory" << ar_status.message();
//	throw "Error occurred in reserving memory";
//  }
//  for (int i = 0; i < end - start; i = i + step) {
//	builder.UnsafeAppend(i);
//  }
//  ar_status = builder.Finish(&index_arr);
//
//  if (!ar_status.ok()) {
//	LOG(ERROR) << "Error occurred in finalizing range index value array";
//  }
//}

//cylon::RangeIndex::RangeIndex(int start, int size, int step, arrow::MemoryPool *pool) : BaseIndex(0, size, pool),
//																						start_(start),
//																						end_(size),
//																						step_(step) {
//
//}
//Status RangeIndex::LocationByValue(const void *search_param,
//								   const std::shared_ptr<arrow::Table> &input,
//								   std::vector<int64_t> &filter_locations,
//								   std::shared_ptr<arrow::Table> &output) {
//  //LOG(INFO) << "Extract From Range Index";
//  arrow::Status arrow_status;
//  cylon::Status status;
//  std::shared_ptr<arrow::Array> out_idx;
//  arrow::compute::ExecContext fn_ctx(GetPool());
//  arrow::Int64Builder idx_builder(GetPool());
//  const arrow::Datum input_table(input);
//
//  status = LocationByValue(search_param, filter_locations);
//
//  if (!status.is_ok()) {
//	LOG(ERROR) << "Error occurred in obtaining filter indices by index value";
//	return status;
//  }
//
//  arrow_status = idx_builder.AppendValues(filter_locations);
//  if (!arrow_status.ok()) {
//	LOG(ERROR) << "Failed appending filter locations";
//	RETURN_CYLON_STATUS_IF_ARROW_FAILED(arrow_status);
//  }
//
//  arrow_status = idx_builder.Finish(&out_idx);
//
//  if (!arrow_status.ok()) {
//	LOG(ERROR) << "Failed index array builder finish";
//	RETURN_CYLON_STATUS_IF_ARROW_FAILED(arrow_status);
//  }
//  const arrow::Datum filter_indices(out_idx);
//  arrow::Result<arrow::Datum>
//	  result = arrow::compute::Take(input_table, filter_indices, arrow::compute::TakeOptions::Defaults(), &fn_ctx);
//  if (!result.ok()) {
//	LOG(ERROR) << "Failed in filtering table by filter indices";
//	RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
//  }
//  output = result.ValueOrDie().table();
//  return Status::OK();
//}
//
//int RangeIndex::GetColId() const {
//  return BaseIndex::GetColId();
//}
//int RangeIndex::GetSize() const {
//  return BaseIndex::GetSize();
//}
//arrow::MemoryPool *RangeIndex::GetPool() const {
//  return BaseIndex::GetPool();
//}
//int RangeIndex::GetStart() const {
//  return start_;
//}
//int RangeIndex::GetAnEnd() const {
//  return end_;
//}
//int RangeIndex::GetStep() const {
//  return step_;
//}
//Status RangeIndex::LocationByValue(const void *search_param, std::vector<int64_t> &find_index) {
//  int64_t val = *((int64_t *)search_param);
//  if (!(val >= start_ && val < end_)) {
//	LOG(ERROR) << "0:Invalid Key, it must be in the range of 0, num of records: " << val << "," << start_ << ","
//			   << end_;
//	return Status(cylon::Code::KeyError);
//  }
//  find_index.push_back(val);
//  return Status::OK();
//}
//Status RangeIndex::LocationByValue(const void *search_param, int64_t &find_index) {
//  int64_t val = *((int64_t *)search_param);
//  if (!(val >= start_ && val < end_)) {
//	LOG(ERROR) << "1:Invalid Key, it must be in the range of 0, num of records";
//	return Status(cylon::Code::KeyError);
//  }
//  find_index = val;
//  return Status::OK();
//}
//std::shared_ptr<arrow::Array> RangeIndex::GetIndexAsArray() {
//  auto pool = GetPool();
//  BuildRangeIndexArray(start_, end_, step_, pool, index_arr_);
//  return index_arr_;
//}
//void RangeIndex::SetIndexArray(std::shared_ptr<arrow::Array> &index_arr) {
//  index_arr_ = index_arr;
//}
//std::shared_ptr<arrow::Array> RangeIndex::GetIndexArray() {
//  if (index_arr_ == nullptr) {
//	index_arr_ = GetIndexAsArray();
//  }
//  return index_arr_;
//}
//bool RangeIndex::IsUnique() {
//  return true;
//}
//IndexingSchema RangeIndex::GetSchema() {
//  return Range;
//}

//RangeIndexKernel::RangeIndexKernel() {}

//std::shared_ptr<BaseIndex> RangeIndexKernel::BuildIndex(arrow::MemoryPool *pool,
//														std::shared_ptr<arrow::Table> &input_table,
//														const int index_column) {
//  std::shared_ptr<RangeIndex> range_index;
//
//  range_index = std::make_shared<RangeIndex>(0, input_table->num_rows(), 1, pool);
//
//  return range_index;
//}
//
//int BaseIndex::GetColId() const {
//  return col_id_;
//}
//int BaseIndex::GetSize() const {
//  return size_;
//}
//arrow::MemoryPool *BaseIndex::GetPool() const {
//  return pool_;
//}

LinearArrowIndexKernel::LinearArrowIndexKernel() {}

std::shared_ptr<BaseArrowIndex> LinearArrowIndexKernel::BuildIndex(arrow::MemoryPool *pool,
																   std::shared_ptr<arrow::Table> &input_table,
																   const int index_column) {

  std::shared_ptr<arrow::Array> index_array;

  if (input_table->column(0)->num_chunks() > 1) {
	const arrow::Result<std::shared_ptr<arrow::Table>> &res = input_table->CombineChunks(pool);
	if (!res.status().ok()) {
	  LOG(ERROR) << "Error occurred in combining chunks in table";
	}
	input_table = res.ValueOrDie();
  }

  index_array = cylon::util::GetChunkOrEmptyArray(input_table->column(index_column), 0);
  auto
	  index =
	  std::make_shared<ArrowLinearIndex>(index_column, input_table->num_rows(), pool, index_array);

  return index;
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
	if (cast_val->Equals(val)) {
	  find_index.push_back(ix);
	}
  }
  return Status::OK();
}
Status ArrowLinearIndex::LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param, int64_t &find_index) {
  auto cast_val = search_param->CastTo(index_array_->type()).ValueOrDie();
  for (int64_t ix = 0; ix < index_array_->length(); ix++) {
	auto val = index_array_->GetScalar(ix).ValueOrDie();
	if (cast_val->Equals(val)) {
	  find_index = ix;
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
  const arrow::Datum input_table(input);

  status = LocationByValue(search_param, filter_location);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in obtaining filter indices by index value";
	return status;
  }

  arrow_status = idx_builder.AppendValues(filter_location);

  if (!arrow_status.ok()) {
	LOG(ERROR) << "Error occurred in appending filter indices to builder";
	RETURN_CYLON_STATUS_IF_ARROW_FAILED(arrow_status);
  }

  arrow_status = idx_builder.Finish(&out_idx);

  if (!arrow_status.ok()) {
	LOG(ERROR) << "Error occurred in builder finish";
	RETURN_CYLON_STATUS_IF_ARROW_FAILED(arrow_status);
  }

  const arrow::Datum filter_indices(out_idx);
  arrow::Result<arrow::Datum>
	  result = arrow::compute::Take(input_table, filter_indices, arrow::compute::TakeOptions::Defaults(), &fn_ctx);

  if (!result.status().ok()) {
	LOG(ERROR) << "Error occurred in filtering table by indices";
	RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
  }

  output = result.ValueOrDie().table();
  return Status::OK();
}
std::shared_ptr<arrow::Array> ArrowLinearIndex::GetIndexAsArray() {
  return index_array_;
}
void ArrowLinearIndex::SetIndexArray(std::shared_ptr<arrow::Array> &index_arr) {
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
IndexingSchema ArrowLinearIndex::GetSchema() {
  return Linear;
}
arrow::MemoryPool *ArrowLinearIndex::GetPool() const {
  return BaseArrowIndex::GetPool();
}
bool ArrowLinearIndex::IsUnique() {
  bool is_unique = false;
  auto status = CompareArraysForUniqueness(index_array_, is_unique);
  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in is unique operation";
  }
  return is_unique;
}
Status ArrowLinearIndex::LocationByVector(const std::shared_ptr<arrow::Array> &search_param,
										  std::vector<int64_t> &filter_location) {

  auto cast_search_param_result = arrow::compute::Cast(search_param, index_array_->type());

  auto search_param_array = cast_search_param_result.ValueOrDie().make_array();

  auto res_isin_filter = arrow::compute::IsIn(index_array_, search_param_array);

  if (!res_isin_filter.ok()) {
	LOG(ERROR) << "Filtering Failed when creating resultant index bool array!!!";
  }

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
	LOG(ERROR) << "1:Invalid Key, it must be in the range of 0, num of records";
	return Status(cylon::Code::KeyError);
  }
  find_index.push_back(val);
  return Status::OK();
}
Status ArrowRangeIndex::LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param, int64_t &find_index) {
  std::shared_ptr<arrow::Int64Scalar> casted_search_param = std::static_pointer_cast<arrow::Int64Scalar>(search_param);
  int64_t val = casted_search_param->value;
  if (!(val >= start_ && val < end_)) {
	LOG(ERROR) << "1:Invalid Key, it must be in the range of 0, num of records";
	return Status(cylon::Code::KeyError);
  }
  find_index = val;
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
  const arrow::Datum input_table(input);

  status = LocationByValue(search_param, filter_location);

  if (!status.is_ok()) {
	LOG(ERROR) << "Error occurred in obtaining filter indices by index value";
	return status;
  }

  arrow_status = idx_builder.AppendValues(filter_location);
  if (!arrow_status.ok()) {
	LOG(ERROR) << "Failed appending filter locations";
	RETURN_CYLON_STATUS_IF_ARROW_FAILED(arrow_status);
  }

  arrow_status = idx_builder.Finish(&out_idx);

  if (!arrow_status.ok()) {
	LOG(ERROR) << "Failed index array builder finish";
	RETURN_CYLON_STATUS_IF_ARROW_FAILED(arrow_status);
  }
  const arrow::Datum filter_indices(out_idx);
  arrow::Result<arrow::Datum>
	  result = arrow::compute::Take(input_table, filter_indices, arrow::compute::TakeOptions::Defaults(), &fn_ctx);
  if (!result.ok()) {
	LOG(ERROR) << "Failed in filtering table by filter indices";
	RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
  }
  output = result.ValueOrDie().table();
  return Status::OK();
}
std::shared_ptr<arrow::Array> ArrowRangeIndex::GetIndexAsArray() {
  arrow::Status arrow_status;
  auto pool = GetPool();

  arrow::Int64Builder builder(pool);

  std::shared_ptr<arrow::Int64Array> index_array;

  std::vector<int64_t> vec(GetSize(), 1);

  for (int64_t ix = 0; ix < GetSize(); ix += GetStep()) {
	vec[ix] = ix;
  }

  builder.AppendValues(vec);
  arrow_status = builder.Finish(&index_array);

  if (!arrow_status.ok()) {
	LOG(ERROR) << "Error occurred in retrieving index";
	return nullptr;
  }

  return index_array;
}
void ArrowRangeIndex::SetIndexArray(std::shared_ptr<arrow::Array> &index_arr) {
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
IndexingSchema ArrowRangeIndex::GetSchema() {
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

  for(int64_t ix=0; ix < cast_search_param->length() ; ix++) {
    int64_t index_value = cast_search_param->Value(ix);
    filter_location.push_back(index_value);
  }

  return Status::OK();
};
ArrowRangeIndexKernel::ArrowRangeIndexKernel() {

}
std::shared_ptr<BaseArrowIndex> ArrowRangeIndexKernel::BuildIndex(arrow::MemoryPool *pool,
																  std::shared_ptr<arrow::Table> &input_table,
																  const int index_column) {
  std::shared_ptr<ArrowRangeIndex> range_index;
  range_index = std::make_shared<ArrowRangeIndex>(0, input_table->num_rows(), 1, pool);
  return range_index;
}
}








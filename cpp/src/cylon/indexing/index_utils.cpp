#include "index_utils.hpp"
#include "../util/arrow_utils.hpp"

cylon::Status cylon::IndexUtil::BuildArrowHashIndex(const std::shared_ptr<Table> &input,
													const int index_column,
													std::shared_ptr<cylon::BaseArrowIndex> &index) {

  auto table_ = input->get_table();
  auto ctx = input->GetContext();
  if (table_->column(0)->num_chunks() > 1) {
	const arrow::Result<std::shared_ptr<arrow::Table>> &res = table_->CombineChunks(cylon::ToArrowPool(ctx));
	RETURN_CYLON_STATUS_IF_ARROW_FAILED(res.status());
	table_ = res.ValueOrDie();
  }
  auto pool = cylon::ToArrowPool(ctx);
  std::shared_ptr<cylon::ArrowIndexKernel> kernel = CreateArrowIndexKernel(table_, index_column);
  RETURN_CYLON_STATUS_IF_FAILED(kernel->BuildIndex(pool, table_, index_column, index));
  auto index_array = cylon::util::GetChunkOrEmptyArray(table_->column(index_column), 0);
  return cylon::Status::OK();
}

cylon::Status cylon::IndexUtil::BuildArrowIndex(const cylon::IndexingType schema,
												const std::shared_ptr<Table> &input,
												const int index_column,
												std::shared_ptr<cylon::BaseArrowIndex> &index) {
  switch (schema) {
	case Range: return BuildArrowRangeIndex(input, index);
	case Linear: return BuildArrowLinearIndex(input, index_column, index);
	case Hash: return BuildArrowHashIndex(input, index_column, index);
	case BinaryTree:break;
	case BTree:break;
	default: BuildArrowRangeIndex(input, index);
  }
  return cylon::Status(cylon::Code::Invalid, "Invalid indexing schema");
}

cylon::Status cylon::IndexUtil::BuildArrowIndexFromArray(const cylon::IndexingType schema,
														 const std::shared_ptr<Table> &input,
														 const std::shared_ptr<arrow::Array> &index_array,
														 std::shared_ptr<Table> &output) {
  cylon::Status status;
  std::shared_ptr<cylon::BaseArrowIndex> index;
  auto ctx = input->GetContext();
  auto pool = cylon::ToArrowPool(ctx);
  switch (schema) {
	case Range: status = BuildArrowRangeIndexFromArray(index_array->length(), pool, index);
	  break;
	case Linear:
	  status = BuildArrowLinearIndexFromArrowArray(const_cast<std::shared_ptr<arrow::Array> &>(index_array),
												   pool,
												   index);
	  break;
	case Hash:
	  status = BuildArrowHashIndexFromArray(const_cast<std::shared_ptr<arrow::Array> &>(index_array), pool, index);
	  break;
	case BinaryTree:status = cylon::Status(cylon::Code::NotImplemented, "Not Implemented");
	  break;
	case BTree:status = cylon::Status(cylon::Code::NotImplemented, "Not Implemented");
	  break;
  }
  RETURN_CYLON_STATUS_IF_FAILED(status);
  RETURN_CYLON_STATUS_IF_FAILED(input->Set_ArrowIndex(index, false));
  output = std::move(input);
  return cylon::Status::OK();
}

cylon::Status cylon::IndexUtil::BuildArrowIndex(cylon::IndexingType schema,
												const std::shared_ptr<Table> &input,
												int index_column,
												bool drop,
												std::shared_ptr<Table> &output) {
  std::shared_ptr<cylon::BaseArrowIndex> index;
  RETURN_CYLON_STATUS_IF_FAILED(BuildArrowIndex(schema, input, index_column, index));
  RETURN_CYLON_STATUS_IF_FAILED(input->Set_ArrowIndex(index, drop));
  output = std::move(input);
  return cylon::Status::OK();
}

cylon::Status cylon::IndexUtil::BuildArrowLinearIndex(const std::shared_ptr<Table> &input,
													  const int index_column,
													  std::shared_ptr<cylon::BaseArrowIndex> &index) {
  auto table_ = input->get_table();
  auto ctx = input->GetContext();
  auto pool = cylon::ToArrowPool(ctx);
  std::shared_ptr<cylon::ArrowIndexKernel> kernel = std::make_shared<cylon::LinearArrowIndexKernel>();
  RETURN_CYLON_STATUS_IF_FAILED(kernel->BuildIndex(pool, table_, index_column, index));
  return cylon::Status::OK();
}
cylon::Status cylon::IndexUtil::BuildArrowRangeIndex(const std::shared_ptr<Table> &input,
													 std::shared_ptr<cylon::BaseArrowIndex> &index) {

  auto ctx = input->GetContext();
  auto pool = cylon::ToArrowPool(ctx);
  auto table_ = input->get_table();
  std::shared_ptr<cylon::ArrowIndexKernel> kernel = std::make_unique<ArrowRangeIndexKernel>();
  RETURN_CYLON_STATUS_IF_FAILED(kernel->BuildIndex(pool, table_, 0, index));
  std::vector<int64_t> range_index_values(input->Rows());
  std::shared_ptr<arrow::Array> index_arr;
  for (int i = 0; i < input->Rows(); ++i) {
	range_index_values.push_back(i);
  }
  arrow::Int64Builder builder(pool);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.Reserve(input->Rows()));
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.AppendValues(range_index_values));
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.Finish(&index_arr));
  index->SetIndexArray(index_arr);
  return cylon::Status::OK();
}
cylon::Status cylon::IndexUtil::BuildArrowRangeIndexFromArray(int64_t size,
															  arrow::MemoryPool *pool,
															  std::shared_ptr<cylon::BaseArrowIndex> &index) {
  index = std::make_shared<ArrowRangeIndex>(0, size, 1, pool);
  return Status::OK();
}
cylon::Status cylon::IndexUtil::BuildArrowHashIndexFromArray(std::shared_ptr<arrow::Array> &index_values,
															 arrow::MemoryPool *pool,
															 std::shared_ptr<cylon::BaseArrowIndex> &index) {
  Status s;
  switch (index_values->type()->id()) {

	case arrow::Type::NA:break;
	case arrow::Type::BOOL:
	  return cylon::IndexUtil::BuildArrowHashIndexFromArrowArray<arrow::BooleanType>(index_values,
																					 pool,
																					 index);
	case arrow::Type::UINT8:
	  return cylon::IndexUtil::BuildArrowHashIndexFromArrowArray<arrow::UInt8Type>(index_values,
																				   pool,
																				   index);
	case arrow::Type::INT8:
	  return cylon::IndexUtil::BuildArrowHashIndexFromArrowArray<arrow::Int8Type>(index_values,
																				  pool,
																				  index);
	case arrow::Type::UINT16:
	  return cylon::IndexUtil::BuildArrowHashIndexFromArrowArray<arrow::UInt16Type>(index_values,
																					pool,
																					index);
	case arrow::Type::INT16:
	  return cylon::IndexUtil::BuildArrowHashIndexFromArrowArray<arrow::Int16Type>(index_values,
																				   pool,
																				   index);
	case arrow::Type::UINT32:
	  return cylon::IndexUtil::BuildArrowHashIndexFromArrowArray<arrow::UInt32Type>(index_values,
																					pool,
																					index);
	case arrow::Type::INT32:
	  return cylon::IndexUtil::BuildArrowHashIndexFromArrowArray<arrow::Int32Type>(index_values,
																				   pool,
																				   index);
	case arrow::Type::UINT64:
	  return cylon::IndexUtil::BuildArrowHashIndexFromArrowArray<arrow::UInt64Type>(index_values,
																					pool,
																					index);
	case arrow::Type::INT64:
	  return cylon::IndexUtil::BuildArrowHashIndexFromArrowArray<arrow::Int64Type>(index_values,
																				   pool,
																				   index);
	case arrow::Type::HALF_FLOAT:
	  return cylon::IndexUtil::BuildArrowHashIndexFromArrowArray<arrow::HalfFloatType>(index_values,
																					   pool,
																					   index);
	case arrow::Type::FLOAT:
	  return cylon::IndexUtil::BuildArrowHashIndexFromArrowArray<arrow::FloatType>(index_values,
																				   pool,
																				   index);
	case arrow::Type::DOUBLE:
	  return cylon::IndexUtil::BuildArrowHashIndexFromArrowArray<arrow::DoubleType>(index_values,
																					pool,
																					index);
	case arrow::Type::STRING:
	  return cylon::IndexUtil::BuildArrowHashIndexFromArrowArray<arrow::StringType, arrow::util::string_view>(
		  index_values,
		  pool,
		  index);;
	case arrow::Type::BINARY:break;
	case arrow::Type::FIXED_SIZE_BINARY:break;
	case arrow::Type::DATE32:break;
	case arrow::Type::DATE64:break;
	case arrow::Type::TIMESTAMP:break;
	case arrow::Type::TIME32:break;
	case arrow::Type::TIME64:break;
	case arrow::Type::INTERVAL_MONTHS:break;
	case arrow::Type::INTERVAL_DAY_TIME:break;
	case arrow::Type::DECIMAL:break;
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
  return cylon::Status(cylon::Code::Invalid, "Unsupported data type");
}
















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

#include <cylon/indexing/index_utils.hpp>
#include <cylon/util/arrow_utils.hpp>

cylon::Status cylon::IndexUtil::BuildArrowHashIndex(const std::shared_ptr<Table> &input,
													const int index_column,
													std::shared_ptr<cylon::BaseArrowIndex> &index) {

  auto table_ = input->get_table();
  const auto& ctx = input->GetContext();
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
														 const std::shared_ptr<arrow::Array> &index_array) {
  cylon::Status status;
  std::shared_ptr<cylon::BaseArrowIndex> index;
  const auto& ctx = input->GetContext();
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
  RETURN_CYLON_STATUS_IF_FAILED(input->SetArrowIndex(index, false));
  return cylon::Status::OK();
}

cylon::Status cylon::IndexUtil::BuildArrowIndex(cylon::IndexingType schema,
												const std::shared_ptr<Table> &input,
												int index_column,
												bool drop,
												std::shared_ptr<Table> &output) {
  std::shared_ptr<cylon::BaseArrowIndex> index;
  RETURN_CYLON_STATUS_IF_FAILED(BuildArrowIndex(schema, input, index_column, index));
  RETURN_CYLON_STATUS_IF_FAILED(input->SetArrowIndex(index, drop));
  output = std::move(input);
  return cylon::Status::OK();
}

cylon::Status cylon::IndexUtil::BuildArrowLinearIndex(const std::shared_ptr<Table> &input,
													  const int index_column,
													  std::shared_ptr<cylon::BaseArrowIndex> &index) {
  auto table_ = input->get_table();
  const auto& ctx = input->GetContext();
  auto pool = cylon::ToArrowPool(ctx);
  std::shared_ptr<cylon::ArrowIndexKernel> kernel = std::make_shared<cylon::LinearArrowIndexKernel>();
  RETURN_CYLON_STATUS_IF_FAILED(kernel->BuildIndex(pool, table_, index_column, index));
  return cylon::Status::OK();
}
cylon::Status cylon::IndexUtil::BuildArrowRangeIndex(const std::shared_ptr<Table> &input,
													 std::shared_ptr<cylon::BaseArrowIndex> &index) {

  const auto& ctx = input->GetContext();
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
cylon::Status cylon::IndexUtil::BuildArrowHashIndexFromArray(const std::shared_ptr<arrow::Array> &index_values,
															 arrow::MemoryPool *pool,
															 std::shared_ptr<cylon::BaseArrowIndex> &index) {
  switch (index_values->type()->id()) {

	case arrow::Type::NA:break;
	case arrow::Type::BOOL:
	  return cylon::IndexUtil::BuildArrowNumericHashIndexFromArrowArray<arrow::BooleanType>(index_values,
																							pool,
																							index);
	case arrow::Type::UINT8:
	  return cylon::IndexUtil::BuildArrowNumericHashIndexFromArrowArray<arrow::UInt8Type>(index_values,
																						  pool,
																						  index);
	case arrow::Type::INT8:
	  return cylon::IndexUtil::BuildArrowNumericHashIndexFromArrowArray<arrow::Int8Type>(index_values,
																						 pool,
																						 index);
	case arrow::Type::UINT16:
	  return cylon::IndexUtil::BuildArrowNumericHashIndexFromArrowArray<arrow::UInt16Type>(index_values,
																						   pool,
																						   index);
	case arrow::Type::INT16:
	  return cylon::IndexUtil::BuildArrowNumericHashIndexFromArrowArray<arrow::Int16Type>(index_values,
																						  pool,
																						  index);
	case arrow::Type::UINT32:
	  return cylon::IndexUtil::BuildArrowNumericHashIndexFromArrowArray<arrow::UInt32Type>(index_values,
																						   pool,
																						   index);
	case arrow::Type::INT32:
	  return cylon::IndexUtil::BuildArrowNumericHashIndexFromArrowArray<arrow::Int32Type>(index_values,
																						  pool,
																						  index);
	case arrow::Type::UINT64:
	  return cylon::IndexUtil::BuildArrowNumericHashIndexFromArrowArray<arrow::UInt64Type>(index_values,
																						   pool,
																						   index);
	case arrow::Type::INT64:
	  return cylon::IndexUtil::BuildArrowNumericHashIndexFromArrowArray<arrow::Int64Type>(index_values,
																						  pool,
																						  index);
	case arrow::Type::HALF_FLOAT:
	  return cylon::IndexUtil::BuildArrowNumericHashIndexFromArrowArray<arrow::HalfFloatType>(index_values,
																							  pool,
																							  index);
	case arrow::Type::FLOAT:
	  return cylon::IndexUtil::BuildArrowNumericHashIndexFromArrowArray<arrow::FloatType>(index_values,
																						  pool,
																						  index);
	case arrow::Type::DOUBLE:
	  return cylon::IndexUtil::BuildArrowNumericHashIndexFromArrowArray<arrow::DoubleType>(index_values,
																						   pool,
																						   index);
	case arrow::Type::STRING:
	  return cylon::IndexUtil::BuildArrowBinaryHashIndexFromArrowArray<arrow::StringType>(
		  index_values,
		  pool,
		  index);
	case arrow::Type::BINARY:
	  return cylon::IndexUtil::BuildArrowBinaryHashIndexFromArrowArray<arrow::BinaryType>(
		  index_values,
		  pool,
		  index);
	case arrow::Type::FIXED_SIZE_BINARY:
	  return cylon::IndexUtil::BuildArrowBinaryHashIndexFromArrowArray<arrow::BinaryType>(
		  index_values,
		  pool,
		  index);
	case arrow::Type::DATE32:
	  return cylon::IndexUtil::BuildArrowNumericHashIndexFromArrowArray<arrow::Date32Type>(index_values,
																						   pool,
																						   index);
	case arrow::Type::DATE64:
	  return cylon::IndexUtil::BuildArrowNumericHashIndexFromArrowArray<arrow::Date64Type>(index_values,
																						   pool,
																						   index);
	case arrow::Type::TIMESTAMP:
	  return cylon::IndexUtil::BuildArrowNumericHashIndexFromArrowArray<arrow::TimestampType>(index_values,
																							  pool,
																							  index);
	case arrow::Type::TIME32:
	  return cylon::IndexUtil::BuildArrowNumericHashIndexFromArrowArray<arrow::Time32Type>(index_values,
																						   pool,
																						   index);
	case arrow::Type::TIME64:
	  return cylon::IndexUtil::BuildArrowNumericHashIndexFromArrowArray<arrow::Time64Type>(index_values,
																						   pool,
																						   index);
	default: return cylon::Status(cylon::Code::Invalid, "Unsupported data type");
  }
  return cylon::Status(cylon::Code::Invalid, "Unsupported data type");
}
















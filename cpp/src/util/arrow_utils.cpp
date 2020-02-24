#include <arrow/compute/api.h>
#include <arrow/api.h>
#include "arrow_utils.h"

namespace twisterx::util {

arrow::Status sort_column(const std::shared_ptr<arrow::Array> &data_column,
						  std::shared_ptr<arrow::DataType> type,
						  const std::shared_ptr<arrow::Int64Array> &sorted_indices,
						  std::shared_ptr<arrow::Array> sorted_data_column) {
  int64_t length = sorted_indices->length();

#define NUMERIC_ARRAY_SORT(TYPE) \
{\
  arrow::NumericBuilder<TYPE> array_builder; \
  arrow::Status reserveStatus = array_builder.Reserve(length); \
  auto casted_data_array = std::static_pointer_cast<arrow::NumericArray<TYPE>>(data_column); \
  for (int64_t index = 0; index < length; ++index) { \
    int64_t current_index = sorted_indices->Value(index); \
    arrow::Status status = array_builder.Append(casted_data_array->Value(current_index)); \
    if (status != arrow::Status::OK()) { \
      return status; \
    } \
  } \
  return array_builder.Finish(&sorted_data_column);\
}

  switch (type->id()) {
	case arrow::Type::UINT8: NUMERIC_ARRAY_SORT(arrow::UInt8Type)
	case arrow::Type::NA:break;
	case arrow::Type::BOOL:break;
	case arrow::Type::INT8: NUMERIC_ARRAY_SORT(arrow::Int8Type);
	case arrow::Type::UINT16:break;
	case arrow::Type::INT16:break;
	case arrow::Type::UINT32:break;
	case arrow::Type::INT32:break;
	case arrow::Type::UINT64:break;
	case arrow::Type::INT64:break;
	case arrow::Type::HALF_FLOAT:break;
	case arrow::Type::FLOAT: NUMERIC_ARRAY_SORT(arrow::FloatType);
	case arrow::Type::DOUBLE:break;
	case arrow::Type::STRING:break;
	case arrow::Type::BINARY:break;
	case arrow::Type::FIXED_SIZE_BINARY:break;
	case arrow::Type::DATE32:break;
	case arrow::Type::DATE64:break;
	case arrow::Type::TIMESTAMP:break;
	case arrow::Type::TIME32:break;
	case arrow::Type::TIME64:break;
	case arrow::Type::INTERVAL:break;
	case arrow::Type::DECIMAL:break;
	case arrow::Type::LIST:break;
	case arrow::Type::STRUCT:break;
	case arrow::Type::UNION:break;
	case arrow::Type::DICTIONARY:break;
	case arrow::Type::MAP:break;
	case arrow::Type::EXTENSION:break;
	case arrow::Type::FIXED_SIZE_LIST:break;
	case arrow::Type::DURATION:break;
	case arrow::Type::LARGE_STRING:break;
	case arrow::Type::LARGE_BINARY:break;
	case arrow::Type::LARGE_LIST:break;
  }
#undef NUMERIC_ARRAY_SORT
}

std::shared_ptr<arrow::Table> sort_table(std::shared_ptr<arrow::Table> tab, int64_t sort_column_index,
										 arrow::MemoryPool *memory_pool) {
  std::shared_ptr<arrow::Table> tab_to_process;
  // combine chunks if multiple chunks are available
  if (tab->column(sort_column_index)->num_chunks() > 1) {
	arrow::Status left_combine_stat = tab->CombineChunks(memory_pool, &tab);
  } else {
	tab_to_process = tab;
  }
  auto column_to_sort = tab_to_process->column(sort_column_index)->chunk(0);

  // sort to indices
  std::shared_ptr<arrow::Array> sorted_column_index;
  arrow::compute::FunctionContext ctx;
  arrow::Status status = arrow::compute::SortToIndices(&ctx, *column_to_sort, &sorted_column_index);
  auto index_lookup = std::static_pointer_cast<arrow::Int64Array>(sorted_column_index);

  // now sort everything based on sorted index
  std::vector<std::shared_ptr<arrow::Array>> sorted_columns;
  int64_t no_of_columns = tab_to_process->num_columns();
  for (int64_t col_index = 0; col_index < no_of_columns; ++col_index) {
	std::shared_ptr<arrow::Array> sorted_array;
	sort_column(tab_to_process->column(col_index)->chunk(0),
				tab_to_process->column(0)->type(),
				index_lookup,
				sorted_array);

  }
  return arrow::Table::Make(tab->schema(), sorted_columns);
}
}
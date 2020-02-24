#include <arrow/compute/api.h>
#include <arrow/array/builder_base.h>
#include <arrow/api.h>
#include "arrow_utils.h"

namespace twisterx::util {

arrow::Status copy_int64_by_index(int64_t index,
								  const std::shared_ptr<arrow::Array>& data_column,
								  const std::shared_ptr<arrow::ArrayBuilder>& array_builder) {
  auto casted_builder = std::static_pointer_cast<arrow::NumericBuilder<arrow::Int64Type>>(array_builder);
  auto casted_data_array = std::static_pointer_cast<arrow::Int64Array>(data_column);
  return casted_builder->Append(casted_data_array->Value(index));
}

void sort_column(const std::shared_ptr<arrow::Array> &data_column,
				 const std::shared_ptr<arrow::Int64Array> &sorted_indices,
				 const std::shared_ptr<arrow::ArrayBuilder> &array_builder) {

  int64_t length = sorted_indices->length();
  arrow::Status reserveStatus = array_builder->Reserve(length);
  for (int64_t index = 0; index < length; ++index) {
	int64_t current_index = sorted_indices->Value(index);
	copy_int64_by_index(current_index, data_column, array_builder);
  }
}

template<typename JOIN_COLUMN_ARRAY, typename ARROW_KEY_TYPE, typename CPP_KEY_TYPE>
void sort_table(std::shared_ptr<arrow::Table> tab, int64_t sort_column_index,
				const std::shared_ptr<std::unordered_map<int64_t, std::shared_ptr<arrow::ArrayBuilder>>>& column_builders,
				arrow::MemoryPool *memory_pool) {
  std::shared_ptr<arrow::Table> tab_to_process;
  // combine chunks if multiple chunks are available
  if (tab->column(sort_column_index)->num_chunks() > 1) {
	arrow::Status left_combine_stat = tab->CombineChunks(memory_pool, &tab);
  } else {
	tab_to_process = tab;
  }
  auto column_to_sort = std::static_pointer_cast<JOIN_COLUMN_ARRAY>(
	  tab_to_process->column(sort_column_index)->chunk(0));

  // sort to indices
  std::shared_ptr<arrow::Array> sorted_column_index;
  arrow::compute::FunctionContext ctx;
  arrow::Status status = arrow::compute::SortToIndices(&ctx, *column_to_sort, &sorted_column_index);
  auto index_lookup = std::static_pointer_cast<arrow::Int64Array>(sorted_column_index);

  // now sort everything based on sorted index
  int64_t no_of_columns = tab_to_process->num_columns();
  for (int64_t col_index = 0; col_index < no_of_columns; ++col_index) {
	sort_column(tab_to_process->column(col_index)->chunk(0), index_lookup,
				std::shared_ptr<arrow::ArrayBuilder>(column_builders->find(col_index)->second));
  }
}
}
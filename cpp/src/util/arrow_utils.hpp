#ifndef TWISTERX_SRC_UTIL_ARROW_UTILS_HPP_
#define TWISTERX_SRC_UTIL_ARROW_UTILS_HPP_
#include <arrow/table.h>

namespace twisterx {
	namespace util {
		arrow::Status sort_table(std::shared_ptr<arrow::Table> tab, int64_t sort_column_index,
														 std::shared_ptr<arrow::Table> *sorted_table,
														 arrow::MemoryPool *memory_pool = arrow::default_memory_pool());

		arrow::Status copy_array_by_indices(std::shared_ptr<std::vector<int64_t>> indices,
																				std::shared_ptr<arrow::Array> source_array,
																				std::shared_ptr<arrow::Array> *copied_array,
																				arrow::MemoryPool *memory_pool = arrow::default_memory_pool());
		/**
		 * Free the buffers of a arrow table, after this, the table is no-longer valid
		 * @param table the table pointer
		 * @return if success
		 */
		arrow::Status free_table(const std::shared_ptr<arrow::Table>& table);
	}
}
#endif //TWISTERX_SRC_UTIL_ARROW_UTILS_HPP_

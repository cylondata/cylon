#ifndef TWISTERX_SRC_UTIL_ARROW_UTILS_HPP_
#define TWISTERX_SRC_UTIL_ARROW_UTILS_HPP_
#include <arrow/table.h>

namespace twisterx {
namespace util {
std::shared_ptr<arrow::Table> sort_table(std::shared_ptr<arrow::Table> tab, int64_t sort_column_index,
										 arrow::MemoryPool *memory_pool = arrow::default_memory_pool());

std::shared_ptr<arrow::Array> copy_array_by_indices(const std::shared_ptr<std::vector<int64_t>> &indices,
													const std::shared_ptr<arrow::Array>& source_array,
													arrow::MemoryPool *memory_pool = arrow::default_memory_pool());
}
}
#endif //TWISTERX_SRC_UTIL_ARROW_UTILS_HPP_

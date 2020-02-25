#ifndef TWISTERX_SRC_UTIL_ARROW_UTILS_H_
#define TWISTERX_SRC_UTIL_ARROW_UTILS_H_
#include <arrow/table.h>

namespace twisterx::util {
std::shared_ptr<arrow::Table> sort_table(std::shared_ptr<arrow::Table> tab, int64_t sort_column_index,
				arrow::MemoryPool *memory_pool);
std::shared_ptr<arrow::Table> sort_table(std::shared_ptr<arrow::Table> tab, int64_t sort_column_index);
}
#endif //TWISTERX_SRC_UTIL_ARROW_UTILS_H_

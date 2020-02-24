#ifndef TWISTERX_SRC_UTIL_ARROW_UTILS_H_
#define TWISTERX_SRC_UTIL_ARROW_UTILS_H_
#include <arrow/table.h>

namespace twisterx::util {
template<typename JOIN_COLUMN_ARRAY, typename ARROW_KEY_TYPE, typename CPP_KEY_TYPE>
void sort_table(std::shared_ptr<arrow::Table> tab, int64_t sort_column_index,
				std::shared_ptr<std::unordered_map<int64_t, arrow::ArrayBuilder>> column_builders,
				arrow::MemoryPool *memory_pool);
}
#endif //TWISTERX_SRC_UTIL_ARROW_UTILS_H_

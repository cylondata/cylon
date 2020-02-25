#ifndef TWISTERX_SRC_JOIN_JOIN_UTILS_HPP_
#define TWISTERX_SRC_JOIN_JOIN_UTILS_HPP_

#include <arrow/api.h>
#include <map>

namespace twisterx::join::util {
std::shared_ptr<arrow::Table> build_final_table(const std::shared_ptr<std::map<int64_t,
																			   std::vector<int64_t >>> &joined_indices,
												const std::shared_ptr<arrow::Table> &left_tab,
												const std::shared_ptr<arrow::Table> &right_tab,
												arrow::MemoryPool *memory_pool);
}
#endif //TWISTERX_SRC_JOIN_JOIN_UTILS_HPP_

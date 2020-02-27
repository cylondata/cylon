#ifndef TWISTERX_SRC_JOIN_JOIN_UTILS_HPP_
#define TWISTERX_SRC_JOIN_JOIN_UTILS_HPP_

#include <arrow/api.h>
#include <map>

namespace twisterx {
	namespace join {
		namespace util {
			arrow::Status build_final_table(const std::shared_ptr<std::map<int64_t,
          std::shared_ptr<std::vector<int64_t>>>> &joined_indices,
																			const std::shared_ptr<arrow::Table> &left_tab,
																			const std::shared_ptr<arrow::Table> &right_tab,
																			std::shared_ptr<arrow::Table> *final_table,
																			arrow::MemoryPool *memory_pool);
		}
	}
}
#endif //TWISTERX_SRC_JOIN_JOIN_UTILS_HPP_

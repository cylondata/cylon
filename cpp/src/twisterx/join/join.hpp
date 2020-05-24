#ifndef TWISTERX_TX_JOIN_H
#define TWISTERX_TX_JOIN_H

#include <arrow/table.h>
#include "../arrow/arrow_kernels.hpp"
#include "../arrow/arrow_hash_kernels.hpp"
#include "join_config.h"

namespace twisterx {
namespace join {

arrow::Status joinTables(const std::shared_ptr<arrow::Table> &left_tab,
						 const std::shared_ptr<arrow::Table> &right_tab,
						 twisterx::join::config::JoinConfig join_config,
						 std::shared_ptr<arrow::Table> *joined_table,
						 arrow::MemoryPool *memory_pool = arrow::default_memory_pool());

arrow::Status joinTables(const std::vector<std::shared_ptr<arrow::Table>> &left_tabs,
						 const std::vector<std::shared_ptr<arrow::Table>> &right_tabs,
						 twisterx::join::config::JoinConfig join_config,
						 std::shared_ptr<arrow::Table> *joined_table,
						 arrow::MemoryPool *memory_pool = arrow::default_memory_pool());

}
}
#endif //TWISTERX_TX_JOIN_H

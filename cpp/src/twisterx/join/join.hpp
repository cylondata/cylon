#ifndef TWISTERX_TX_JOIN_H
#define TWISTERX_TX_JOIN_H

#include <arrow/table.h>
#include "../arrow/arrow_kernels.hpp"
#include "join_config.h"

namespace twisterx {
namespace join {

arrow::Status joinTables(const std::shared_ptr<arrow::Table> &left_tab,
                         const std::shared_ptr<arrow::Table> &right_tab,
                         int64_t left_join_column_idx,
                         int64_t right_join_column_idx,
                         twisterx::join::config::JoinType join_type,
                         twisterx::join::config::JoinAlgorithm join_algorithm,
                         std::shared_ptr<arrow::Table> *joined_table,
                         arrow::MemoryPool *memory_pool = arrow::default_memory_pool());

arrow::Status joinTables(const std::vector<std::shared_ptr<arrow::Table>> &left_tabs,
                         const std::vector<std::shared_ptr<arrow::Table>> &right_tabs,
                         int64_t left_join_column_idx,
                         int64_t right_join_column_idx,
                         twisterx::join::config::JoinType join_type,
                         twisterx::join::config::JoinAlgorithm join_algorithm,
                         std::shared_ptr<arrow::Table> *joined_table,
                         arrow::MemoryPool *memory_pool = arrow::default_memory_pool());

}
}
#endif //TWISTERX_TX_JOIN_H

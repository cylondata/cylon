#ifndef TWISTERX_TX_JOIN_H
#define TWISTERX_TX_JOIN_H

#include <arrow/table.h>

namespace twisterx { namespace join {

enum JoinType {
  LEFT, RIGHT, INNER
};

enum JoinAlgorithm {
  SORT, HASH
};

std::shared_ptr<arrow::Table> join(const std::shared_ptr<arrow::Table> &left_tab,
								   const std::shared_ptr<arrow::Table> &right_tab,
								   int64_t left_join_column_idx,
								   int64_t right_join_column_idx,
								   JoinType join_type,
								   JoinAlgorithm join_algorithm,
								   arrow::MemoryPool *memory_pool);

std::shared_ptr<arrow::Table> join(const std::vector<std::shared_ptr<arrow::Table>> &left_tabs,
								   const std::vector<std::shared_ptr<arrow::Table>> &right_tabs,
								   int64_t left_join_column_idx,
								   int64_t right_join_column_idx,
								   JoinType join_type,
								   JoinAlgorithm join_algorithm,
								   arrow::MemoryPool *memory_pool);
} }
#endif //TWISTERX_TX_JOIN_H

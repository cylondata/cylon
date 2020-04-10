#ifndef TWISTERX_SRC_IO_TABLE_API_H_
#define TWISTERX_SRC_IO_TABLE_API_H_
#include <string>
#include <vector>
#include "status.hpp"
#include <arrow/api.h>

namespace twisterx {

void put_table(const std::string &id, const std::shared_ptr<arrow::Table> &table);

twisterx::Status read_csv(const std::string &path, const std::string &id);
twisterx::Status joinTables(const std::string &table_left,
                            const std::string &table_right,
                            int left_col_idx,
                            int right_col_idx,
                            const std::string &dest_id);
int column_count(const std::string &id);
int row_count(const std::string &id);
twisterx::Status print(const std::string &table_id, int col1, int col2, int row1, int row2);
twisterx::Status merge(std::vector<std::string> table_ids, const std::string& merged_tab);

/**
 * Sort the table with the given identifier
 * @param id table id
 * @param columnIndex the sorting column index
 * @return the sorted table
 */
twisterx::Status sortTable(const std::string& tableId, const std::string& sortTableId, int columnIndex);

/**
 * Partition the table into multiple tables
 * @param id
 * @param hash_columns
 * @param no_of_partitions
 * @param out
 * @param pool
 * @return
 */
twisterx::Status hashPartition(const std::string& id, const std::vector<int>& hash_columns, int no_of_partitions,
                               std::vector<std::shared_ptr<arrow::Table>> *out, arrow::MemoryPool *pool);
}
#endif //TWISTERX_SRC_IO_TABLE_API_H_

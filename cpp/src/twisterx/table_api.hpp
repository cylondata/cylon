#ifndef TWISTERX_SRC_IO_TABLE_API_H_
#define TWISTERX_SRC_IO_TABLE_API_H_

#include <string>
#include <vector>
#include "status.cpp"
#include "join/join_config.h"
#include "io/csv_read_config.h"
#include <arrow/api.h>

namespace twisterx {

    std::shared_ptr<arrow::Table> get_table(const std::string &id);

    void put_table(const std::string &id, const std::shared_ptr<arrow::Table> &table);

    twisterx::Status from_csv(const std::string &path, const std::string &id, const char delimiter);

    twisterx::Status read_csv(const std::string &path, const std::string &id,
                              twisterx::io::config::CSVReadOptions options = twisterx::io::config::CSVReadOptions());

    twisterx::Status JoinTables(const std::string &table_left,
                                const std::string &table_right,
                                twisterx::join::config::JoinConfig join_config,
                                const std::string &dest_id);

    int column_count(const std::string &id);

    int row_count(const std::string &id);

/**
 * Print a table
 * @param table_id id of the table
 * @param col1
 * @param col2
 * @param row1
 * @param row2
 * @return
 */
    twisterx::Status print(const std::string &table_id, int col1, int col2, int row1, int row2);

    twisterx::Status print_to_ostream(const std::string &table_id,
                                      int col1,
                                      int col2,
                                      int row1,
                                      int row2,
                                      std::ostream &out);

/**
 * Merge the set of tables into a single table, each table should have the same schema
 *
 * @param table_ids ids of the tables
 * @param merged_tab id of the merged table
 * @return the status of the merge
 */
    twisterx::Status merge(std::vector<std::string> table_ids, const std::string &merged_tab);

/**
 * Sort the table with the given identifier
 * @param id table id
 * @param columnIndex the sorting column index
 * @return the status of the merge
 */
    twisterx::Status sortTable(const std::string &tableId, const std::string &sortTableId, int columnIndex);

/**
 * Partition the table into multiple tables using a hash function, hash will be applied to the bytes of the data
 * @param id the table id
 * @param hash_columns the hash columns, at the moment we only use the first column
 * @param no_of_partitions number of partitions to output
 * @param out tables created after hashing
 * @param pool the memory pool
 * @return the status of the partition operation
 */
    twisterx::Status hashPartition(const std::string &id, const std::vector<int> &hash_columns, int no_of_partitions,
                                   std::vector<std::shared_ptr<arrow::Table>> *out, arrow::MemoryPool *pool);
}
#endif //TWISTERX_SRC_IO_TABLE_API_H_

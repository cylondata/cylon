#ifndef TWISTERX_SRC_IO_TABLE_H_
#define TWISTERX_SRC_IO_TABLE_H_

#include <memory>
#include <string>
#include <vector>
#include <glog/logging.h>

#include "status.hpp"
#include "util/uuid.h"
#include "column.hpp"

namespace twisterx {

/**
 * Table provides the main API for using TwisterX for data processing.
 */
class Table {

 public:
  /**
   * Tables can only be created using the factory methods, so the constructor is private
   */
  Table(std::string id) {
    id_ = id;
  }

  /**
   * Create a table by reading a csv file
   * @param path file path
   * @return a pointer to the table
   */
  static Status FromCSV(const std::string &path, std::unique_ptr<Table> *tableOut);

  /**
   * Create table from parquet
   * @param path file path
   * @return a pointer to the table
   */
  static Status FromParquet(const std::string &path, std::shared_ptr<Table> out);

  /**
   * Create a table from set of columns
   * @param columns the columns
   * @return the created table
   */
  static Status FromColumns(std::vector<std::shared_ptr<Column>> columns, std::shared_ptr<Table> out);

  /**
   * Create a table from an arrow table,
   * @param table
   * @return
   */
  static Status FromArrowTable(std::shared_ptr<arrow::Table> table);

  /**
   * Write the table as a CSV
   * @param path file path
   * @return the status of the operation
   */
  Status WriteCSV(const std::string &path);

  /**
   * Write the table to parquet file
   * @param path file path
   * @return status of the operation
   */
  Status write_parquet(const std::string &path);

  /**
   * Partition the table based on the hash
   * @param hash_columns the columns use for has
   * @param no_of_partitions number partitions
   * @return new set of tables each with the new partition
   */
  Status HashPartition(const std::vector<int>& hash_columns, int no_of_partitions, std::vector<std::shared_ptr<twisterx::Table>> *out);

  /**
   * Merge the set of tables to create a single table
   * @param tables
   * @return new merged table
   */
  static Status Merge(const std::vector<std::shared_ptr<twisterx::Table>>& tables, std::unique_ptr<Table> *tableOut);

  /**
   * Sort the table according to the given column, this is a local sort
   * @param sort_column
   * @return new table sorted according to the sort column
   */
  Status Sort(int sort_column, std::unique_ptr<Table> *tableOut);

  /**
   *
   * @param right
   * @return
   */
  std::shared_ptr<Table> join(Table right);

  /*END OF TRANSFORMATION FUNCTIONS*/
  int columns();

  int rows();

  void clear();

  void print();

  void print(int row1, int row2, int col1, int col2);

  std::string get_id() {
    return this->id_;
  }

private:
  /**
   * Every table should have an unique id
   */
  std::string id_;
};
}

#endif //TWISTERX_SRC_IO_TABLE_H_

#ifndef TWISTERX_SRC_IO_TABLE_H_
#define TWISTERX_SRC_IO_TABLE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <glog/logging.h>
#include "io/csv_read_config.h"

#include "status.hpp"
#include "util/uuid.hpp"
#include "column.hpp"
#include "join/join_config.h"
#include "arrow/arrow_join.hpp"
#include "join/join.hpp"

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
	id_ = std::move(id);
  }

  /**
   * Create a table by reading a csv file
   * @param path file path
   * @return a pointer to the table
   */
  static Status FromCSV(const std::string &path,
						std::shared_ptr<Table> *tableOut,
						twisterx::io::config::CSVReadOptions options = twisterx::io::config::CSVReadOptions());

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
   * Partition the table based on the hash
   * @param hash_columns the columns use for has
   * @param no_of_partitions number partitions
   * @return new set of tables each with the new partition
   */
  Status HashPartition(const std::vector<int> &hash_columns,
					   int no_of_partitions,
					   std::vector<std::shared_ptr<twisterx::Table>> *out);

  /**
   * Merge the set of tables to create a single table
   * @param tables
   * @return new merged table
   */
  static Status
  Merge(const std::vector<std::shared_ptr<twisterx::Table>> &tables, std::unique_ptr<Table> *tableOut);

  /**
   * Sort the table according to the given column, this is a local sort
   * @param sort_column
   * @return new table sorted according to the sort column
   */
  Status Sort(int sort_column, std::unique_ptr<Table> *tableOut);

  /**
   * Do the join with the right table
   * @param right the right table
   * @param joinConfig the join configurations
   * @param out the final table
   * @return success
   */
  Status Join(const std::shared_ptr<Table> &right,
			  twisterx::join::config::JoinConfig join_config,
			  std::shared_ptr<Table> *out);

  /**
   * Create a arrow table from this data structure
   * @param out arrow table
   * @return the status of the operation
   */
  Status ToArrowTable(std::shared_ptr<arrow::Table> *out);

  /*END OF TRANSFORMATION FUNCTIONS*/
  int columns();

  int rows();

  void clear();

  void show();

  void show(int row1, int row2, int col1, int col2);

  std::string get_id() {
	return this->id_;
  }

  static Status from_csv(const std::string &path, const char &delimiter, const std::string &uuid);

 private:
  /**
   * Every table should have an unique id
   */
  std::string id_;
};
}

#endif //TWISTERX_SRC_IO_TABLE_H_

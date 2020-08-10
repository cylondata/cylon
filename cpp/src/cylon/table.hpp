/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CYLON_SRC_IO_TABLE_H_
#define CYLON_SRC_IO_TABLE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <glog/logging.h>
#include "io/csv_read_config.hpp"

#include "status.hpp"
#include "util/uuid.hpp"
#include "column.hpp"
#include "join/join_config.hpp"
#include "arrow/arrow_join.hpp"
#include "join/join.hpp"
#include "io/csv_write_config.hpp"
#include "row.hpp"

namespace cylon {

/**
 * Table provides the main API for using cylon for data processing.
 */
class Table {
 public:
  /**
   * Tables can only be created using the factory methods, so the constructor is private
   */
  Table(std::shared_ptr<arrow::Table> &tab, cylon::CylonContext *ctx) {
    this->table_ = tab;
    this->ctx = ctx;
  }

  virtual ~Table();

  /**
   * Create a table by reading a csv file
   * @param path file path
   * @return a pointer to the table
   */
  static Status FromCSV(cylon::CylonContext *ctx, const std::string &path,
                        std::shared_ptr<Table> &tableOut,
                        const cylon::io::config::CSVReadOptions &options = cylon::io::config::CSVReadOptions());

  /**
   * Read multiple CSV files into multiple tables. If threading is enabled, the tables will be read
   * in parallel
   * @param ctx
   * @param paths
   * @param tableOuts
   * @param options
   * @return
   */
  static Status FromCSV(cylon::CylonContext *ctx, const std::vector<std::string> &paths,
                        const std::vector<std::shared_ptr<Table> *> &tableOuts,
                        io::config::CSVReadOptions options = cylon::io::config::CSVReadOptions());

  /**
   * Create a table from an arrow table,
   * @param table
   * @return
   */
  static Status FromArrowTable(cylon::CylonContext *ctx,
                               std::shared_ptr<arrow::Table> &table,
                               std::shared_ptr<Table> *tableOut);

  /**
   * Write the table as a CSV
   * @param path file path
   * @return the status of the operation
   */
  Status WriteCSV(const std::string &path,
                  const cylon::io::config::CSVWriteOptions &options = cylon::io::config::CSVWriteOptions());

  /**
   * Create a arrow table from this data structure
   * @param output arrow table
   * @return the status of the operation
   */
  Status ToArrowTable(std::shared_ptr<arrow::Table> &output);

  /**
   * Partition the table based on the hash
   * @param hash_columns the columns use for has
   * @param no_of_partitions number partitions
   * @return new set of tables each with the new partition
   */
  Status HashPartition(const std::vector<int> &hash_columns,
                       int no_of_partitions,
                       std::unordered_map<int, std::shared_ptr<cylon::Table>> *output);

  /**
   * Merge the set of tables to create a single table
   * @param tables
   * @return new merged table
   */
  static Status Merge(cylon::CylonContext *ctx,
                      const std::vector<std::shared_ptr<cylon::Table>> &tables,
                      shared_ptr<Table> &tableOut);

  /**
   * Sort the table according to the given column, this is a local sort
   * @param sort_column
   * @return new table sorted according to the sort column
   */
  Status Sort(int sort_column, shared_ptr<Table> &output);

  /**
   * Do the join with the right table
   * @param right the right table
   * @param joinConfig the join configurations
   * @param output the final table
   * @return success
   */
  static Status Join(std::shared_ptr<Table> &left, std::shared_ptr<Table> &right,
              cylon::join::config::JoinConfig join_config,
              std::shared_ptr<Table> *output);

  /**
   * Similar to local join, but performs the join in a distributed fashion
   * @param right
   * @param join_config
   * @param output
   * @return <cylon::Status>
   */
  static Status DistributedJoin(std::shared_ptr<Table> &left, std::shared_ptr<Table> &right,
                         cylon::join::config::JoinConfig join_config,
                         std::shared_ptr<Table> *output);

  /**
   * Performs union with the passed table
   * @param other right table
   * @param output output table
   * @return <cylon::Status>
   */
  static Status Union(std::shared_ptr<Table> &first, std::shared_ptr<Table> &second,
      std::shared_ptr<Table> &output);

  /**
   * Similar to local union, but performs the union in a distributed fashion
   * @param other
   * @param output
   * @return
   */
  static Status DistributedUnion(std::shared_ptr<Table> &left, std::shared_ptr<Table> &right,
                                 std::shared_ptr<Table> &out);

  /**
   * Performs subtract/difference with the passed table
   * @param right right table
   * @param output output table
   * @return <cylon::Status>
   */
  static Status Subtract(std::shared_ptr<Table> &first,
                         std::shared_ptr<Table> &second, std::shared_ptr<Table> &out);

  /**
   * Similar to local subtract/difference, but performs in a distributed fashion
   * @param other
   * @param output
   * @return
   */
  static Status DistributedSubtract(std::shared_ptr<Table> &left, std::shared_ptr<Table> &right,
                                    std::shared_ptr<Table> &out);

  /**
   * Performs intersection with the passed table
   * @param other right table
   * @param output output table
   * @return <cylon::Status>
   */
  static Status Intersect(std::shared_ptr<Table> &first,
                          std::shared_ptr<Table> &second, std::shared_ptr<Table> &output);

  /**
   * Similar to local intersection, but performs in a distributed fashion
   * @param other
   * @param output
   * @return
   */
  static Status DistributedIntersect(std::shared_ptr<Table> &left, std::shared_ptr<Table> &right,
                                     std::shared_ptr<Table> &out);

  /**
   * Filters out rows based on the selector function
   * @param selector lambda function returning a bool
   * @param output
   * @return
   */
  Status Select(const std::function<bool(cylon::Row)> &selector, std::shared_ptr<Table> &output);

  /**
   * Creates a View of an existing table by dropping one or more columns
   * @param project_columns
   * @param output
   * @return
   */
  Status Project(const std::vector<int64_t> &project_columns, std::shared_ptr<Table> &output);

  /**
   * Print the col range and row range
   * @param col1 start col
   * @param col2 end col
   * @param row1 start row
   * @param row2 end row
   * @param out the stream
   * @param delimiter delimiter between values
   * @param use_custom_header custom header
   * @param headers the names of custom header
   * @return true if print is successful
   */
  Status PrintToOStream(
      int col1,
      int col2,
      int row1,
      int row2,
      std::ostream &out,
      char delimiter = ',',
      bool use_custom_header = false,
      const std::vector<std::string> &headers = {});

  /*END OF TRANSFORMATION FUNCTIONS*/

  /**
   * Get the number of columns in the table
   * @return numbre of columns
   */
  int32_t Columns();

  /**
   * Get the number of rows in this table
   * @return number of rows in the table
   */
  int64_t Rows();

  /**
   * Print the complete table
   */
  void Print();

  /**
   * Print the table from row1 to row2 and col1 to col2
   * @param row1 first row to start printing (including)
   * @param row2 end row to stop printing (including)
   * @param col1 first column to start printing (including)
   * @param col2 end column to stop printing (including)
   */
  void Print(int row1, int row2, int col1, int col2);

  /**
   * Get the underlying arrow table
   * @return the arrow table
   */
  shared_ptr<arrow::Table> get_table();

  /**
   * Clears the table
   */
  void Clear();

  /**
   * Returns the cylon Context
   * @return
   */
  cylon::CylonContext *GetContext();

  /**
   * Get column names of the table
   * @return vector<string>
   */
  std::vector<std::string> ColumnNames();

  /**
   * Set to true to free the memory of this table when it is not needed
   */
  void retainMemory(bool retain) {
    retain_ = retain;
  }

  bool IsRetain() const;
 private:
  /**
   * Every table should have an unique id
   */
  std::string id_;
  cylon::CylonContext *ctx;
  std::shared_ptr<arrow::Table> table_;
  bool retain_ = true;
};
}  // namespace cylon



#endif //CYLON_SRC_IO_TABLE_H_

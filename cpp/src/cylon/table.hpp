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
  Table(std::string id, cylon::CylonContext *ctx) {
    id_ = std::move(id);
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

  static Status FromCSV(cylon::CylonContext *ctx, const std::vector<std::string> &paths,
                        const std::vector<std::shared_ptr<Table> *> &tableOuts,
                        const cylon::io::config::CSVReadOptions &options = cylon::io::config::CSVReadOptions());

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
  static Status FromArrowTable(const std::shared_ptr<arrow::Table> &table);

  static Status FromArrowTable(cylon::CylonContext *ctx,
                               const std::shared_ptr<arrow::Table> &table,
                               std::shared_ptr<Table> *tableOut);

  Status Project(const std::vector<int64_t> &project_columns, std::shared_ptr<Table> &out);

  /**
   * Write the table as a CSV
   * @param path file path
   * @return the status of the operation
   */
  Status WriteCSV(const std::string &path,
                  const cylon::io::config::CSVWriteOptions &options = cylon::io::config::CSVWriteOptions());

  /**
   * Partition the table based on the hash
   * @param hash_columns the columns use for has
   * @param no_of_partitions number partitions
   * @return new set of tables each with the new partition
   */
  Status HashPartition(const std::vector<int> &hash_columns,
                       int no_of_partitions,
                       std::vector<std::shared_ptr<cylon::Table>> *out);

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
  Status Sort(int sort_column, shared_ptr<Table> &out);

  /**
   * Do the join with the right table
   * @param right the right table
   * @param joinConfig the join configurations
   * @param out the final table
   * @return success
   */
  Status Join(const std::shared_ptr<Table> &right,
              cylon::join::config::JoinConfig join_config,
              std::shared_ptr<Table> *out);

  Status DistributedJoin(const shared_ptr<Table> &right,
                         cylon::join::config::JoinConfig join_config,
                         std::shared_ptr<Table> *out);

  Status Union(const std::shared_ptr<Table> &right, std::shared_ptr<Table> &out);

  Status DistributedUnion(const shared_ptr<Table> &right, shared_ptr<Table> &out);

  Status Subtract(const std::shared_ptr<Table> &right, std::shared_ptr<Table> &out);

  Status DistributedSubtract(const shared_ptr<Table> &right, shared_ptr<Table> &out);

  Status Intersect(const std::shared_ptr<Table> &right, std::shared_ptr<Table> &out);

  Status DistributedIntersect(const shared_ptr<Table> &right, shared_ptr<Table> &out);

  Status Select(const std::function<bool(cylon::Row)> &selector, std::shared_ptr<Table> &out);

  /**
   * Create a arrow table from this data structure
   * @param out arrow table
   * @return the status of the operation
   */
  Status ToArrowTable(std::shared_ptr<arrow::Table> &out);

  /*END OF TRANSFORMATION FUNCTIONS*/
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
   * Get the id associated with this table
   * @return string id
   */
  std::string GetID() {
    return this->id_;
  }

  void Clear();

  cylon::CylonContext *GetContext();

 private:
  /**
   * Every table should have an unique id
   */
  std::string id_;

  cylon::CylonContext *ctx;

  typedef Status(*SetOperation)
      (CylonContext *ctx, const std::string &left_table, const std::string &right_table, const std::string &out_table);
  Status DoSetOperation(SetOperation operation, const shared_ptr<Table> &right, shared_ptr<Table> &out);

};
}

#endif //CYLON_SRC_IO_TABLE_H_

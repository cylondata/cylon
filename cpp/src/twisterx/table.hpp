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
#include "io/csv_write_config.h"
#include "row.hpp"

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
                        const twisterx::io::config::CSVReadOptions &options = twisterx::io::config::CSVReadOptions());

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

  static Status FromArrowTable(std::shared_ptr<arrow::Table> table, std::shared_ptr<Table> *tableOut);

  /**
   * Write the table as a CSV
   * @param path file path
   * @return the status of the operation
   */
  Status WriteCSV(const std::string &path,
                  const twisterx::io::config::CSVWriteOptions &options = twisterx::io::config::CSVWriteOptions());

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
  static Status Merge(const std::vector<std::shared_ptr<twisterx::Table>> &tables,
                      std::shared_ptr<Table> *tableOut);

  /**
   * Sort the table according to the given column, this is a local sort
   * @param sort_column
   * @return new table sorted according to the sort column
   */
  Status Sort(int sort_column, std::shared_ptr<Table> *tableOut);

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

  Status DistributedJoin(twisterx::TwisterXContext *ctx,
                         const std::shared_ptr<Table> &right,
                         twisterx::join::config::JoinConfig join_config,
                         std::shared_ptr<Table> *out);

  Status Union(const std::shared_ptr<Table> &right, std::shared_ptr<Table> &out);

  Status DistributedUnion(twisterx::TwisterXContext *ctx, const std::shared_ptr<Table> &right, std::shared_ptr<Table> &out);

  Status Select(const std::function<bool(twisterx::Row)> &selector, std::shared_ptr<Table> &out);

  /**
   * Create a arrow table from this data structure
   * @param out arrow table
   * @return the status of the operation
   */
  Status ToArrowTable(std::shared_ptr<arrow::Table> &out);

  /*END OF TRANSFORMATION FUNCTIONS*/
  int32_t columns();

  /**
   * Get the number of rows in this table
   * @return number of rows in the table
   */
  int64_t rows();

  /**
   * Print the complete table
   */
  void print();

  /**
   * Print the table from row1 to row2 and col1 to col2
   * @param row1 first row to start printing (including)
   * @param row2 end row to stop printing (including)
   * @param col1 first column to start printing (including)
   * @param col2 end column to stop printing (including)
   */
  void print(int row1, int row2, int col1, int col2);

  /**
   * Get the id associated with this table
   * @return string id
   */
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

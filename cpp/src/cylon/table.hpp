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

#include <glog/logging.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "indexing/index.hpp"
#include "io/csv_read_config.hpp"

#ifdef BUILD_CYLON_PARQUET
#include "io/parquet_config.hpp"
#endif

#include "column.hpp"
#include "ctx/cylon_context.hpp"
#include "io/csv_write_config.hpp"
#include "join/join.hpp"
#include "join/join_config.hpp"
#include "row.hpp"
#include "status.hpp"
#include "util/uuid.hpp"

namespace cylon {

/**
 * Table provides the main API for using cylon for data processing.
 */
class Table {
 public:

  /**
   * Table created from an arrow::Table
   * @param ctx
   * @param tab (shared_ptr is passed by value and the copy is moved as a class member)
   */
  Table(const std::shared_ptr<CylonContext> &ctx, std::shared_ptr<arrow::Table> tab);

  /**
   * Table created from cylon::Column
   * @param ctx
   * @param cols (vector is passed by value and the copy is moved as a class member)
   */
  Table(const std::shared_ptr<CylonContext> &ctx, std::vector<std::shared_ptr<Column>> cols);

  virtual ~Table();

  /**
   * Create a table from an arrow table,
   * @param table arrow::Table
   * @return
   */
  static Status FromArrowTable(const std::shared_ptr<CylonContext> &ctx,
                               const std::shared_ptr<arrow::Table> &table,
                               std::shared_ptr<Table> &tableOut);

  /**
   * Create a table from cylon columns
   * @param ctx
   * @param columns
   * @param tableOut
   * @return
   */
  static Status FromColumns(const std::shared_ptr<CylonContext> &ctx,
                            const std::vector<std::shared_ptr<Column>> &columns,
                            std::shared_ptr<Table> &tableOut);

  /**
   * Create a arrow table from this data structure
   * @param output arrow table
   * @return the status of the operation
   */
  Status ToArrowTable(std::shared_ptr<arrow::Table> &output);

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
  Status PrintToOStream(int col1, int col2, int row1, int row2, std::ostream &out,
						char delimiter = ',', bool use_custom_header = false,
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
  std::shared_ptr<arrow::Table> get_table();

  /**
   * Clears the table
   */
  void Clear();

  /**
   * Returns the cylon Context
   * @return
   */
  const std::shared_ptr<cylon::CylonContext> &GetContext() const;

  /**
   * Get column names of the table
   * @return vector<string>
   */
  std::vector<std::string> ColumnNames();

  /**
   * Set to true to free the memory of this table when it is not needed
   */
  void retainMemory(bool retain) { retain_ = retain; }

  /**
   * Returns if this table retains data after any operation performed on it
   * @return
   */
  bool IsRetain() const;

  /**
   * Get the i'th column from the table
   * @param index
   * @return
   */
  std::shared_ptr<Column> GetColumn(int32_t index) const;

  /**
   * Get the column vector of the table
   * @return
   */
//  std::vector<std::shared_ptr<cylon::Column>> GetColumns() const;
  const std::vector<std::shared_ptr<cylon::Column>> &GetColumns() const;

  Status SetArrowIndex(std::shared_ptr<cylon::BaseArrowIndex> &index, bool drop_index);

  std::shared_ptr<BaseArrowIndex> GetArrowIndex();

  Status ResetArrowIndex(bool drop = false);

  Status AddColumn(int64_t position, std::string column_name, std::shared_ptr<arrow::Array> &input_column);

 private:
  /**
   * Every table should have an unique id
   */
  const std::shared_ptr<cylon::CylonContext> ctx;
  std::shared_ptr<arrow::Table> table_;
  bool retain_ = true;
  std::vector<std::shared_ptr<cylon::Column>> columns_;
  std::shared_ptr<cylon::BaseArrowIndex> base_arrow_index_ = nullptr;
};

/**
 * Create a table by reading a csv file
 * @param path file path
 * @return a pointer to the table
 */
Status FromCSV(const std::shared_ptr<CylonContext> &ctx, const std::string &path,
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
Status FromCSV(const std::shared_ptr<CylonContext> &ctx, const std::vector<std::string> &paths,
               const std::vector<std::shared_ptr<Table> *> &tableOuts,
               io::config::CSVReadOptions options = cylon::io::config::CSVReadOptions());

/**
 * Write the table as a CSV
 * @param table shared pointer to the cylon table
 * @param path file path
 * @return the status of the operation
 */
Status WriteCSV(const std::shared_ptr<Table> &table, const std::string &path,
                const cylon::io::config::CSVWriteOptions &options = cylon::io::config::CSVWriteOptions());

/**
   * Merge the set of tables to create a single table
   * @param tables
   * @return new merged table
   */
Status Merge(const std::vector<std::shared_ptr<cylon::Table>> &tables, std::shared_ptr<Table> &tableOut);

/**
 * Do the join with the right table
 * @param left the left table
 * @param right the right table
 * @param joinConfig the join configurations
 * @param output the final table
 * @return success
 */
Status Join(std::shared_ptr<Table> &left, std::shared_ptr<Table> &right,
			const join::config::JoinConfig &join_config, std::shared_ptr<Table> &output);

/**
 * Similar to local join, but performs the join in a distributed fashion
 * @param left
 * @param right
 * @param join_config
 * @param output
 * @return <cylon::Status>
 */
Status DistributedJoin(std::shared_ptr<Table> &left, std::shared_ptr<Table> &right,
					   const join::config::JoinConfig &join_config, std::shared_ptr<Table> &output);

/**
 * Performs union with the passed table
 * @param first
 * @param second
 * @param output
 * @return <cylon::Status>
 */
Status Union(const std::shared_ptr<Table> &first, const std::shared_ptr<Table> &second,
			 std::shared_ptr<Table> &output);

/**
 * Similar to local union, but performs the union in a distributed fashion
 * @param first
 * @param second
 * @param output
 * @return <cylon::Status>
 */
Status DistributedUnion(std::shared_ptr<Table> &first, std::shared_ptr<Table> &second,
						std::shared_ptr<Table> &out);

/**
 * Performs subtract/difference with the passed table
 * @param first
 * @param second
 * @param output
 * @return <cylon::Status>
 */
Status Subtract(const std::shared_ptr<Table> &first, const std::shared_ptr<Table> &second,
				std::shared_ptr<Table> &out);

/**
 * Similar to local subtract/difference, but performs in a distributed fashion
 * @param first
 * @param second
 * @param output
 * @return <cylon::Status>
 */
Status DistributedSubtract(std::shared_ptr<Table> &left, std::shared_ptr<Table> &right,
						   std::shared_ptr<Table> &out);

/**
 * Performs intersection with the passed table
 * @param first
 * @param second
 * @param output
 * @return <cylon::Status>
 */
Status Intersect(const std::shared_ptr<Table> &first, const std::shared_ptr<Table> &second,
				 std::shared_ptr<Table> &output);

/**
 * Similar to local intersection, but performs in a distributed fashion
 * @param first
 * @param second
 * @param output
 * @return <cylon::Status>
 */
Status DistributedIntersect(std::shared_ptr<Table> &left, std::shared_ptr<Table> &right,
							std::shared_ptr<Table> &out);

/**
 * Shuffles a table based on hashes
 * @param table
 * @param hash_col_idx vector of column indicies that needs to be hashed
 * @param output
 * @return
 */
Status Shuffle(std::shared_ptr<cylon::Table> &table, const std::vector<int> &hash_col_idx,
			   std::shared_ptr<cylon::Table> &output);

/**
 * Partition the table based on the hash
 * @param hash_columns the columns use for has
 * @param no_of_partitions number partitions
 * @return new set of tables each with the new partition
 */
Status HashPartition(std::shared_ptr<cylon::Table> &table, const std::vector<int> &hash_columns,
					 int no_of_partitions,
					 std::unordered_map<int, std::shared_ptr<cylon::Table>> *output);

/**
 * Sort the table according to the given column, this is a local sort (if the table has chunked
 * columns, they will be merged in the output table)
 * @param sort_column
 * @return new table sorted according to the sort column
 */
Status Sort(std::shared_ptr<cylon::Table> &table, int sort_column, std::shared_ptr<Table> &output,
			bool ascending = true);

/**
 * Sort the table according to the given set of columns, this is a local sort (if the table has
 * chunked columns, they will be merged in the output table)
 * @param sort_column
 * @return new table sorted according to the sort columns
 */
Status Sort(std::shared_ptr<cylon::Table> &table, const std::vector<int32_t> &sort_columns,
			std::shared_ptr<cylon::Table> &out, bool ascending);

/**
 * Sort the table according to the given set of columns and respective ordering direction, this is a
 * local sort (if the table has chunked columns, they will be merged in the output table).
 *
 * Sort direction 'true' indicates ascending ordering and false indicate descending ordering.
 * @param sort_column
 * @return new table sorted according to the sort columns
 */
Status Sort(std::shared_ptr<cylon::Table> &table, const std::vector<int32_t> &sort_columns,
			std::shared_ptr<cylon::Table> &out, const std::vector<bool> &sort_direction);

/**
 * Sort the table according to the given column, this is a local sort (if the table has chunked
 * columns, they will be merged in the output table)
 * @param sort_column
 * @return new table sorted according to the sort column
 */

struct SortOptions {
  uint32_t num_bins;
  uint64_t num_samples;

  static SortOptions Defaults() { return {0, 0}; }
};
Status DistributedSort(std::shared_ptr<cylon::Table> &table,
					   int sort_column,
					   std::shared_ptr<Table> &output,
					   bool ascending = true,
					   SortOptions sort_options = SortOptions::Defaults());

Status DistributedSort(std::shared_ptr<cylon::Table> &table,
					   const std::vector<int> &sort_columns,
					   std::shared_ptr<Table> &output,
					   const std::vector<bool> &sort_direction,
					   SortOptions sort_options = SortOptions::Defaults());

/**
 * Filters out rows based on the selector function
 * @param table
 * @param selector lambda function returning a bool
 * @param output
 * @return
 */
Status Select(std::shared_ptr<cylon::Table> &table, const std::function<bool(cylon::Row)> &selector,
			  std::shared_ptr<Table> &output);

/**
 * Creates a View of an existing table by dropping one or more columns
 * @param table
 * @param project_columns
 * @param output
 * @return
 */
Status Project(std::shared_ptr<cylon::Table> &table, const std::vector<int32_t> &project_columns,
			   std::shared_ptr<Table> &output);

/**
 * Creates a new table by dropping the duplicated elements column-wise
 * @param table
 * @param cols
 * @param out
 * @return Status
 */
Status Unique(std::shared_ptr<cylon::Table> &in, const std::vector<int> &cols,
			  std::shared_ptr<cylon::Table> &out, bool first = true);

Status DistributedUnique(std::shared_ptr<cylon::Table> &in, const std::vector<int> &cols,
						 std::shared_ptr<cylon::Table> &out);

#ifdef BUILD_CYLON_PARQUET
/**
 * Create a table by reading a parquet file
 * @param path file path
 * @return a pointer to the table
 */
Status FromParquet(std::shared_ptr<cylon::CylonContext> &ctx, const std::string &path,
				   std::shared_ptr<Table> &tableOut);
/**
 * Read multiple parquet files into multiple tables. If threading is enabled, the tables will be
 * read in parallel
 * @param ctx
 * @param paths
 * @param tableOuts
 * @param options
 * @return
 */
Status FromParquet(std::shared_ptr<cylon::CylonContext> &ctx, const std::vector<std::string> &paths,
				   const std::vector<std::shared_ptr<Table> *> &tableOuts,
				   io::config::ParquetOptions options = cylon::io::config::ParquetOptions());
/**
 * Write the table as a parquet file
 * @param path file path
 * @return the status of the operation
 */
Status WriteParquet(
	std::shared_ptr<cylon::Table> &table, std::shared_ptr<cylon::CylonContext> &ctx,
	const std::string &path,
	const cylon::io::config::ParquetOptions &options = cylon::io::config::ParquetOptions());
#endif  // BUILD_CYLON_PARQUET

}  // namespace cylon

#endif  // CYLON_SRC_IO_TABLE_H_

---
id: cpp
title: C++
---

## `cylon::CylonContext` 
The entry point to cylon operations

### Initialization 
Local initialization
```c++
  auto ctx = cylon::CylonContext::Init();
```

Distributed initialization
```c++
  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
```

### Context methods

```cpp
  /**
   * Completes and closes all operations under the context
   */
  void Finalize();

  /**
   * Adds a configuration
   * @param <std::string> key
   * @param <std::string> value
   */
  void AddConfig(const std::string &key, const std::string &value);

  /**
   * Returns a configuration
   * @param <std::string> key
   * @param <std::string> def Default value
   * @return <std::string> configuration value
   */
  std::string GetConfig(const std::string &key, const std::string &def = "");

  /**
   * Returns the Communicator instance
   * @return <cylon::net::Communicator>
   */
  net::Communicator *GetCommunicator() const;

  /**
   * Sets a Communicator
   * @param <cylon::net::Communicator*> pointer to another communicator
   */
  void setCommunicator(net::Communicator *communicator1);

  /**
   * Sets if distributed
   * @param <bool> distributed
   */
  void setDistributed(bool distributed);

  /**
   * Returns the local rank
   * @return rank <int>
   */
  int GetRank();

  /**
   * Returns the world size
   * @return world size <int>
   */
  int GetWorldSize();

  /**
   * Returns the neighbors in the world
   * @param include_self
   * @return a std::vector<int> of ranks
   */
  vector<int> GetNeighbours(bool include_self);

  /**
   * Returns memory pool
   * @return <cylon::MemoryPool>
   */
  cylon::MemoryPool *GetMemoryPool();

  /**
   * Sets a memory pool
   * @param <cylon::MemoryPool> mem_pool
   */
  void SetMemoryPool(cylon::MemoryPool *mem_pool);

```

## `cylon::Table` 

### Reading tables 

A `cylon::Table` can be created from a csv file as follows. 

```cpp
std::shared_ptr<cylon::Table> table1;
auto read_options = CSVReadOptions();
auto status = cylon::FromCSV(ctx, "/path/to/csv", table1, read_options))
```

Read a set of tables using threads,

```cpp
std::shared_ptr<cylon::Table> table1, table2;
auto read_options = CSVReadOptions().UseThreads(true);
auto status = cylon::FromCSV(ctx, {"/path/to/csv1.csv", "/path/to/csv2.csv"}, {table1, table2}, read_options);
```

An `arrow::Table` can be imported as follows,

```cpp
std::shared_ptr<cylon::Table> table1;
std::shared_ptr<arrow::Table> some_arrow_table = ...;
auto status = cylon::Table::FromArrowTable(ctx, some_arrow_table, table1);
```

### Writing tables 

A `cylon::Table` can be written to a CSV file as follows, 

```cpp
std::shared_ptr<cylon::Table> table1; 
...
auto write_options = cylon::io::config::CSVWriteOptions();
auto status = WriteCSV(table1, "/path/to/csv", write_options);
```

A `cylon::Table` can be coverted into an `arrow::Table` by simply, 

```cpp
std::shared_ptr<arrow::Table> some_arrow_table;
std::shared_ptr<cylon::Table> table1; 
...
auto status = table1->ToArrowTable(some_arrow_table);
```

### `cylon::Table` API

#### `cylon::Table` Functions 

```cpp
/**
* Create a table from an arrow table,
* @param table
* @return
*/
static Status FromArrowTable(std::shared_ptr<cylon::CylonContext> &ctx,
                           std::shared_ptr<arrow::Table> &table,
                           std::shared_ptr<Table> &tableOut);

/**
* Create a table from cylon columns
* @param ctx
* @param columns
* @param tableOut
* @return
*/
static Status FromColumns(std::shared_ptr<cylon::CylonContext> &ctx,
                        std::vector<std::shared_ptr<Column>> &&columns,
                        std::shared_ptr<Table> &tableOut);

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
Status PrintToOStream(int col1,  int col2,  int row1,  int row2,  std::ostream &out,
                      char delimiter = ',',
                      bool use_custom_header = false,
                      const std::vector<std::string> &headers = {});

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
std::shared_ptr<cylon::CylonContext> GetContext();

/**
* Get column names of the table
* @return vector<string>
*/
std::vector<std::string> ColumnNames();

/**
* Set to true to free the memory of this table when it is not needed
*/
void retainMemory(bool retain);

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
std::vector<std::shared_ptr<cylon::Column>> GetColumns() const;
```

#### `cylon::Table` Operations 
 
```cpp
/**
   * Create a table by reading a csv file
   * @param path file path
   * @return a pointer to the table
   */
Status FromCSV(std::shared_ptr<cylon::CylonContext> &ctx, const std::string &path,
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
Status FromCSV(std::shared_ptr<cylon::CylonContext> &ctx, const std::vector<std::string> &paths,
               const std::vector<std::shared_ptr<Table> *> &tableOuts,
               io::config::CSVReadOptions options = cylon::io::config::CSVReadOptions());

/**
   * Merge the set of tables to create a single table
   * @param tables
   * @return new merged table
   */
Status Merge(std::shared_ptr<cylon::CylonContext> &ctx,
             const std::vector<std::shared_ptr<cylon::Table>> &tables,
             std::shared_ptr<Table> &tableOut);

/**
   * Do the join with the right table
   * @param left the left table
   * @param right the right table
   * @param joinConfig the join configurations
   * @param output the final table
   * @return success
   */
Status Join(std::shared_ptr<Table> &left, std::shared_ptr<Table> &right,
            cylon::join::config::JoinConfig join_config,
            std::shared_ptr<Table> &output);

/**
 * Similar to local join, but performs the join in a distributed fashion
 * @param left
 * @param right
 * @param join_config
 * @param output
 * @return <cylon::Status>
 */
Status DistributedJoin(std::shared_ptr<Table> &left, std::shared_ptr<Table> &right,
                       cylon::join::config::JoinConfig join_config,
                       std::shared_ptr<Table> &output);

/**
 * Performs union with the passed table
 * @param first
 * @param second
 * @param output
 * @return <cylon::Status>
 */
Status Union(std::shared_ptr<Table> &first, std::shared_ptr<Table> &second,
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
Status Subtract(std::shared_ptr<Table> &first,
                std::shared_ptr<Table> &second, std::shared_ptr<Table> &out);

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
Status Intersect(std::shared_ptr<Table> &first,
                 std::shared_ptr<Table> &second, std::shared_ptr<Table> &output);

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
Status HashPartition(std::shared_ptr<cylon::Table> &table,
                     const std::vector<int> &hash_columns,
                     int no_of_partitions,
                     std::unordered_map<int, std::shared_ptr<cylon::Table>> *output);

/**
 * Sort the table according to the given column, this is a local sort (if the table has chunked columns, they will
 * be merged in the output table)
 * @param sort_column
 * @return new table sorted according to the sort column
 */
Status Sort(std::shared_ptr<cylon::Table> &table, int sort_column, std::shared_ptr<Table> &output);

/**
 * Filters out rows based on the selector function
 * @param table
 * @param selector lambda function returning a bool
 * @param output
 * @return
 */
Status Select(std::shared_ptr<cylon::Table> &table, const std::function<bool(cylon::Row)> &selector, std::shared_ptr<Table> &output);

/**
 * Creates a View of an existing table by dropping one or more columns
 * @param table
 * @param project_columns
 * @param output
 * @return
 */
Status Project(std::shared_ptr<cylon::Table> &table, const std::vector<int64_t> &project_columns, std::shared_ptr<Table> &output);
```

### C++ Examples 

Following is a simple C++ API example. 

```cpp
#include <glog/logging.h>

#include <net/mpi/mpi_communicator.hpp>
#include <ctx/cylon_context.hpp>
#include <table.hpp>

#define CHECK_STATUS(status, msg) \
  if (!status.is_ok()) {          \
    LOG(ERROR) << msg << " " << status.get_msg(); \
    ctx->Finalize();              \
    return 1;                     \
  }                               

int main() {

  auto mpi_config = cylon::net::MPIConfig::Make();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
  const int rank = ctx->GetRank() + 1;

  const std::string csv1 = "/tmp/user_device_tm_" + std::to_string(rank) + ".csv";
  const std::string csv2 = "/tmp/user_usage_tm_" + std::to_string(rank) + ".csv";

  std::shared_ptr<cylon::Table> first_table, second_table, joined_table;
  cylon::Status status;

  status = cylon::FromCSV(ctx, csv1, first_table);
  CHECK_STATUS(status, "Reading csv1 failed!")

  status = cylon::FromCSV(ctx, csv2, second_table);
  CHECK_STATUS(status, "Reading csv2 failed!")

  auto join_config = cylon::join::config::JoinConfig::InnerJoin(0, 3);
  status = cylon::DistributedJoin(first_table, second_table, join_config, joined_table);
  CHECK_STATUS(status, "Join failed!")

  LOG(INFO) << "First table had : " << first_table->Rows() << " and Second table had : "
            << second_table->Rows() << ", Joined has : " << joined_table->Rows();

  ctx->Finalize();
  return 0;
}
```

Further examples can be found in [Cylon examples in Github](https://github.com/cylondata/cylon/tree/master/cpp/src/examples).

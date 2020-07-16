---
id: cpp
title: C++ API
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

```c++
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
```c++
std::shared_ptr<cylon::Table> table1;
auto read_options = CSVReadOptions();
auto status = Table::FromCSV(ctx, "/path/to/csv", table1, read_options))
```

Read a set of tables using threads, 
```c++
std::shared_ptr<cylon::Table> table1, table2;
auto read_options = CSVReadOptions().UseThreads(true);
auto status = Table::FromCSV(ctx, {"/path/to/csv1.csv", "/path/to/csv2.csv"}, {table1, table2}, read_options);
```

An `arrow::Table` can be imported as follows, 
```c++
std::shared_ptr<cylon::Table> table1;
std::shared_ptr<arrow::Table> some_arrow_table = ...;
auto status = Table::FromArrowTable(ctx, some_arrow_table, table1);
```

### Writing tables 
A `cylon::Table` can be written to a CSV file as follows, 
```c++
std::shared_ptr<cylon::Table> table1; 
...
auto write_options = cylon::io::config::CSVWriteOptions();
auto status = table1->WriteCSV("/path/to/csv", write_options);
```

A `cylon::Table` can be coverted into an `arrow::Table` by simply, 
```c++
std::shared_ptr<arrow::Table> some_arrow_table;
std::shared_ptr<cylon::Table> table1; 
...
auto status = table1->ToArrowTable(some_arrow_table);
```

### `cylon::Table` Operations
 
```c++
 /**
   * Do the join with the right table
   * @param right the right table
   * @param joinConfig the join configurations
   * @param output the final table
   * @return success
   */
  Status Join(const std::shared_ptr<Table> &right,
              cylon::join::config::JoinConfig join_config,
              std::shared_ptr<Table> *output);

  /**
   * Similar to local join, but performs the join in a distributed fashion
   * @param right
   * @param join_config
   * @param output
   * @return <cylon::Status>
   */
  Status DistributedJoin(const shared_ptr<Table> &right,
                         cylon::join::config::JoinConfig join_config,
                         std::shared_ptr<Table> *output);

  /**
   * Performs union with the passed table
   * @param other right table
   * @param output output table
   * @return <cylon::Status>
   */
  Status Union(const std::shared_ptr<Table> &other, std::shared_ptr<Table> &output);

  /**
   * Similar to local union, but performs the union in a distributed fashion
   * @param other
   * @param output
   * @return
   */
  Status DistributedUnion(const shared_ptr<Table> &other, shared_ptr<Table> &output);

  /**
   * Performs subtract/difference with the passed table
   * @param right right table
   * @param output output table
   * @return <cylon::Status>
   */
  Status Subtract(const std::shared_ptr<Table> &right, std::shared_ptr<Table> &output);

  /**
   * Similar to local subtract/difference, but performs in a distributed fashion
   * @param other
   * @param output
   * @return
   */
  Status DistributedSubtract(const shared_ptr<Table> &right, shared_ptr<Table> &output);

  /**
   * Performs intersection with the passed table
   * @param other right table
   * @param output output table
   * @return <cylon::Status>
   */
  Status Intersect(const std::shared_ptr<Table> &other, std::shared_ptr<Table> &output);

  /**
   * Similar to local intersection, but performs in a distributed fashion
   * @param other
   * @param output
   * @return
   */
  Status DistributedIntersect(const shared_ptr<Table> &other, shared_ptr<Table> &output);

  /**
   * Filters out rows based on the selector function
   * @param selector lambda function returning a bool
   * @param output
   * @return
   */
  Status Select(const std::function<bool(cylon::Row)> &selector, std::shared_ptr<Table> &output);

  /**
   * Creates a simpler view of an existing table by dropping one or more columns
   * @param project_columns
   * @param output
   * @return
   */
  Status Project(const std::vector<int64_t> &project_columns, std::shared_ptr<Table> &output);


  /**
   * Clears the table
   */
  void Clear();
```

### `cylon::Table` attributes 

```c++
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
   * Get the id associated with this table
   * @return string id
   */
  std::string GetID() {
    return this->id_;
  }

  /**
   * Returns the cylon Context
   * @return
   */
  cylon::CylonContext *GetContext();
```
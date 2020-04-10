#ifndef TWISTERX_SRC_IO_TABLE_H_
#define TWISTERX_SRC_IO_TABLE_H_

#include <memory>
#include <string>
#include <vector>

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
   * Create a table by reading a csv file
   * @param path file path
   * @return a pointer to the table
   */
  static std::shared_ptr<Table> from_csv(const std::string &path);

  /**
   * Create table from parquet
   * @param path file path
   * @return a pointer to the table
   */
  static std::shared_ptr<Table> from_parquet(const std::string &path);

  /**
   * Create a table from set of columns
   * @param columns the columns
   * @return the created table
   */
  static std::shared_ptr<Table> from_columns(std::vector<std::shared_ptr<Column>> columns);

  /**
   * Write the table as a CSV
   * @param path file path
   * @return the status of the operation
   */
  Status write_csv(const std::string &path);

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
  std::vector<twisterx::Table> hash_partition(std::vector<int> hash_columns, int no_of_partitions);

  /**
   * Partition round robin
   * @param no_of_partitions
   * @return
   */
  std::vector<twisterx::Table> round_robin_partition(int no_of_partitions);

  /**
   * Merge the set of tables to create a single table
   * @param tables
   * @return new merged table
   */
  static std::shared_ptr<Table> merge(std::vector<std::shared_ptr<twisterx::Table>> tables);

  /**
   * Sort the table according to the given column
   * @param sort_column
   * @return new table sorted according to the sort column
   */
  std::shared_ptr<Table> sort(int sort_column);

  /*END OF TRANSFORMATION FUNCTIONS*/

  int columns();

  int rows();

  void clear();

  void print();

  void print(int row1, int row2, int col1, int col2);

  std::string get_id() {
    return this->id;
  }

 private:
  /**
   * Every table should have an unique id
   */
  std::string id;

  /**
   * Tables can only be created using the factory methods, so the constructor is private
   */
  Table() {
  }

  static std::shared_ptr<Table> create(std::string uuid) {
    std::shared_ptr<Table> t = std::allocate_shared<Table>(A<Table>());
    t->id = uuid;
    return t;
  }

  template<class T>
  struct A : std::allocator<T> {
    void construct(void *p) { ::new(p) Table(); }
    void destroy(Table *p) { p->~Table(); }
  };
};
}

#endif //TWISTERX_SRC_IO_TABLE_H_

#ifndef TWISTERX_SRC_IO_TABLE_H_
#define TWISTERX_SRC_IO_TABLE_H_
#include <arrow/util/string.h>
namespace twisterx {
namespace io {
class Table {
 private:
  std::string id;

  Table(std::string id) {
    this->id = id;
  }

 public:
  /* IO FUNCTIONS*/
  static Table from_csv(const std::string &path);
  static Table from_parquet(const std::string &path);
  static Table write_csv(const std::string &path);
  static Table write_parquet(const std::string &path);
  /* END OF IO FUNCTIONS */

  /*TRANSFORMATION FUNCTIONS*/

  std::vector<twisterx::io::Table> hash_partition(std::vector<int> hash_columns, int no_of_partitions);

  std::vector<twisterx::io::Table> round_robin_partition(int no_of_partitions);

  twisterx::io::Table merge(std::vector<twisterx::io::Table> tables);

  twisterx::io::Table sort(int sort_column);

  /*END OF TRANSFORMATION FUNCTIONS*/

  int columns();

  int rows();

  void clear();

  void print();

  void print(int row1, int row2, int col1, int col2);

  std::string get_id() {
    return this->id;
  }
};
}
}

#endif //TWISTERX_SRC_IO_TABLE_H_

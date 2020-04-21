#ifndef TWISTERX_TABLE_CYTHON_H
#define TWISTERX_TABLE_CYTHON_H

#include "string"
#include "../table.hpp"
#include "../status.hpp"

namespace twisterx {
namespace python {
namespace table {
class CTable {

 private:

  std::string id_;

 public:

  CTable(std::string id);

  std::string get_id();

  int columns();

  int rows();

  void clear();

  void show();

  void show(int row1, int row2, int col1, int col2);

  static Status from_csv(const std::string &path, const char &delimiter, const std::string &uuid);

  Status to_csv(const std::string &path);

  std::string join(const std::string &table_id);

  //unique_ptr<CTable> sort(int sort_column);

};
}
}
}

#endif //TWISTERX_TABLE_CYTHON_H

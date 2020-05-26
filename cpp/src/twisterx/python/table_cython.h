#ifndef TWISTERX_TABLE_CYTHON_H
#define TWISTERX_TABLE_CYTHON_H

#include "string"
#include "../status.hpp"
#include "../join/join_config.h"
#include <arrow/python/pyarrow.h>
#include <arrow/python/serialize.h>
#include "../join/join_config.h"

using namespace twisterx;
using namespace twisterx::join::config;

namespace twisterx {
namespace python {
namespace table {
class CxTable {

 private:

  std::string id_;

 public:

  CxTable(std::string id);

  std::string get_id();

  int columns();

  int rows();

  void clear();

  void show();

  void show(int row1, int row2, int col1, int col2);

  static Status from_csv(const std::string &path, const char &delimiter, const std::string &uuid);

  static std::string from_pyarrow_table(std::shared_ptr<arrow::Table> table);

  static std::shared_ptr<arrow::Table> to_pyarrow_table(const std::string &table_id);

  Status to_csv(const std::string &path);

  std::string join(const std::string &table_id,
				   JoinType type,
				   JoinAlgorithm algorithm,
				   int left_column_index,
				   int right_column_index);

  std::string join(const std::string &table_id, JoinConfig join_config);

  //unique_ptr<CTable> sort(int sort_column);

};
}
}
}

#endif //TWISTERX_TABLE_CYTHON_H

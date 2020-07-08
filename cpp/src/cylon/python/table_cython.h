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

#ifndef CYLON_TABLE_CYTHON_H
#define CYLON_TABLE_CYTHON_H

#include "string"
#include "../status.hpp"
#include <arrow/python/serialize.h>
#include "arrow/api.h"
#include "../join/join_config.hpp"
#include "cylon_context_wrap.h"

using namespace cylon;
using namespace cylon::python;
using namespace cylon::join::config;

namespace cylon {
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

  static cylon::python::cylon_context_wrap* get_new_context();

  static Status from_csv(const std::string &path,
						 const char &delimiter,
						 const std::string &uuid);

  static std::string from_pyarrow_table(std::shared_ptr<arrow::Table> table);

  static std::shared_ptr<arrow::Table> to_pyarrow_table(const std::string &table_id);

  Status to_csv(const std::string &path);

  std::string join(const std::string &table_id,
				   JoinType type,
				   JoinAlgorithm algorithm,
				   int left_column_index,
				   int right_column_index);

  std::string join(const std::string &table_id, JoinConfig join_config);
  //
  std::string distributed_join(const std::string &table_id, JoinConfig join_config);

  std::string distributed_join(cylon_context_wrap *ctx_wrap, std::string &table_id, JoinConfig join_config);

  std::string distributed_join(const std::string &table_id, JoinType type, JoinAlgorithm algorithm, int left_column_index, int right_column_index);

  //unique_ptr<CTable> sort(int sort_column);

};
}
}
}

#endif //CYLON_TABLE_CYTHON_H

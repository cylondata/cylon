#include "table_api.h"
#include <memory>
#include <arrow/api.h>
#include <map>
#include "arrow_io.h"
#include "../join/join.hpp"

namespace twisterx {
namespace io {

std::map<std::string, std::shared_ptr<arrow::Table>> table_map{}; //todo make this un ordered

std::shared_ptr<arrow::Table> get_table(const std::string &id) {
  auto itr = table_map.find(id);
  if (itr != table_map.end()) {
    return itr->second;
  }
  return NULLPTR;
}

void put_table(const std::string &id, const std::shared_ptr<arrow::Table> &table) {
  std::pair<std::string, std::shared_ptr<arrow::Table>> pair(id, table);
  table_map.insert(pair);
}

twisterx::io::Status read_csv(const std::string &path, const std::string &id) {
  arrow::Result<std::shared_ptr<arrow::Table>> result = twisterx::io::read_csv(path);
  if (result.ok()) {
    std::shared_ptr<arrow::Table> table = *result;
    put_table(id, table);
    return twisterx::io::Status(Code::OK, result.status().message());
  }
  return twisterx::io::Status(Code::IOError, result.status().message());;
}

twisterx::io::Status join(const std::string &table_left,
                          const std::string &table_right,
                          int left_col_idx,
                          int right_col_idx,
                          const std::string &dest_id) {
  auto left = get_table(table_left);
  auto right = get_table(table_right);

  if (left == NULLPTR) {
    return twisterx::io::Status(Code::KeyError, "Couldn't find the left table");
  } else if (right == NULLPTR) {
    return twisterx::io::Status(Code::KeyError, "Couldn't find the right table");
  } else {
    std::shared_ptr<arrow::Table> table;
    arrow::Status status = join::join(
        left,
        right,
        left_col_idx,
        right_col_idx,
        join::JoinType::INNER,
        join::JoinAlgorithm::SORT,
        &table,
        arrow::default_memory_pool()
    );
    put_table(dest_id, table);
    return twisterx::io::Status((int) status.code(), status.message());
  }
}

int column_count(const std::string &id) {
  auto table = get_table(id);
  if (table != NULLPTR) {
    return table->num_columns();
  }
  return -1;
}

int row_count(const std::string &id) {
  auto table = get_table(id);
  if (table != NULLPTR) {
    return table->num_rows();
  }
  return -1;
}
}
}
#include "table_api.h"
#include <memory>
#include <arrow/api.h>
#include <map>
#include <iostream>
#include "arrow_io.h"

namespace twisterx {
namespace io {

std::map<std::string, std::shared_ptr<arrow::Table>> table_map{}; //todo make this un ordered

twisterx::io::Status read_csv(const std::string &path, const std::string &id) {
  arrow::Result<std::shared_ptr<arrow::Table>> result = twisterx::io::read_csv(path);
  if (result.ok()) {
    std::shared_ptr<arrow::Table> table = *result;
    std::pair<std::string, std::shared_ptr<arrow::Table>> pair(id, table);
    table_map.insert(pair);
    return twisterx::io::Status(1, result.status().message());
  }
  return twisterx::io::Status(0, result.status().message());;
}

int column_count(const std::string &id) {
  auto itr = table_map.find(id);
  if (itr != table_map.end()) {
    return itr->second->num_columns();
  }
  return -1;
}

int row_count(const std::string &id) {
  auto itr = table_map.find(id);
  if (itr != table_map.end()) {
    return itr->second->num_rows();
  }
  return -1;
}
}
}
#include "table_api.h"
#include <memory>
#include <arrow/api.h>
#include <map>
#include "arrow_io.h"

namespace twisterx {
namespace io {

std::map<std::string, std::shared_ptr<arrow::Table>> table_map{}; //todo make this un ordered

int read_csv(const std::string &path, const std::string &id) {
  std::shared_ptr<arrow::Table> table;
  arrow::Status status = twisterx::io::read_csv(path, table);
  if (status.ok()) {
    std::pair<std::string, std::shared_ptr<arrow::Table>> pair(id, table);
    table_map.insert(pair);
    return 1;
  }
  return 0;
}

int column_count(const std::string &id) {
  auto itr = table_map.find(id);
  if (itr == table_map.end()) {
    return itr->second->num_columns();
  }
  return -1;
}
}
}
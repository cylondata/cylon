#include <map>
#include "table_builder.hpp"
#include <arrow/api.h>
#include "../status.hpp"
#include "../io/arrow_io.hpp"

string twisterx::data::get_id() {
  return "table1";
}

int twisterx::data::get_rows() {
  return 20;
}

int twisterx::data::get_columns() {
  return 10;
}

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

void twisterx::data::read_csv() {
  arrow::Result<std::shared_ptr<arrow::Table>> result = twisterx::io::read_csv("/tmp/csv.csv");
  LOG(INFO) << "Loaded the table " << result.status();
}

#include <arrow/table.h>
#include "table.hpp"
#include "table_api.hpp"
#include "util/uuid.h"

namespace twisterx {

std::shared_ptr<Table> Table::from_csv(const std::string &path) {
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  twisterx::Status status = twisterx::read_csv(path, uuid);
  if (status.is_ok()) {
    return create(uuid);
  }
  throw status.get_msg();
}

int Table::columns() {
  return twisterx::column_count(this->get_id());
}

int Table::rows() {
  return twisterx::row_count(this->get_id());
}

void Table::print() {
  twisterx::print(this->get_id(), 0, this->columns(), 0, this->rows());
}

void Table::print(int row1, int row2, int col1, int col2) {
  twisterx::print(this->get_id(), col1, col2, row1, row2);
}

std::shared_ptr<Table> Table::merge(std::vector<std::shared_ptr<twisterx::Table>> tables) {
  std::vector<std::string> table_ids;
  for (auto it = tables.begin(); it < tables.end(); it++) {
    table_ids.push_back((*it)->get_id());
  }
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  twisterx::Status status = twisterx::merge(table_ids, uuid);
  if (status.is_ok()) {
    return Table::create(uuid);
  } else {
    throw status.get_msg();
  }
}

}

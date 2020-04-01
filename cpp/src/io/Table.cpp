#include "Table.h"
#include "table_api.h"
#include "../util/uuid.h"

namespace twisterx {
namespace io {
Table Table::from_csv(const std::string &path) {
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  twisterx::io::Status status = twisterx::io::read_csv(path, uuid);
  if (status.get_code() == twisterx::io::Code::OK) {
    return Table(uuid);
  }
  throw status.get_msg();
}
int Table::columns() {
  return twisterx::io::column_count(this->get_id());
}
int Table::rows() {
  return twisterx::io::row_count(this->get_id());
}
void Table::print() {
  twisterx::io::print(this->get_id(), 0, this->columns(), 0, this->rows());
}
void Table::print(int row1, int row2, int col1, int col2) {
  twisterx::io::print(this->get_id(), col1, col2, row1, row2);
}
}
}

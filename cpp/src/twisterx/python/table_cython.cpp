#include <fstream>
#include "../table.hpp"
#include "table_cython.h"
#include "../table_api.hpp"

using namespace std;
using namespace twisterx;
using namespace twisterx::python::table;
using namespace twisterx::io::config;
using namespace twisterx::util::uuid;
using namespace twisterx::join::config;

CTable::CTable(std::string id) {
  id_ = id;
}

std::string CTable::get_id() {
  return this->id_;
}

int CTable::columns() {
  return column_count(this->get_id());
}

int CTable::rows() {
  return row_count(this->get_id());
}

void CTable::clear() {

}

void CTable::show() {
  print(this->get_id(), 0, this->columns(), 0, this->rows());
}

void CTable::show(int row1, int row2, int col1, int col2) {
  print(this->get_id(), col1, col2, row1, row2);
}

Status CTable::from_csv(const std::string &path, const char &delimiter, const std::string &uuid) {
  twisterx::Status status = read_csv(path, uuid, CSVReadOptions().WithDelimiter(delimiter));
  return status;
}

Status CTable::to_csv(const std::string &path) {
  ofstream out_csv;
  out_csv.open(path);
  Status status = print_to_ostream(this->get_id(), 0,
								   this->columns(), 0,
								   this->rows(), out_csv);
  out_csv.close();
  return status;
}

std::string CTable::join(const std::string &table_id) {
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  twisterx::Status status = twisterx::JoinTables(
	  this->get_id(),
	  table_id,
	  JoinConfig::RightJoin(0, 1),
	  uuid
  );
  if (status.is_ok()) {
	return uuid;
  } else {
	return "";
  }
}
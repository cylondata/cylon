#include "../table.hpp"
#include "table_cython.h"
#include "../table_api.hpp"

using namespace std;
using namespace twisterx;
using namespace twisterx::python::table;
using namespace twisterx::io::config;

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
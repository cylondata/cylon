#include <fstream>
#include "../table.hpp"
#include "table_cython.h"
#include "../table_api.hpp"
#include <arrow/python/serialize.h>
#include "arrow/api.h"


using namespace std;
using namespace twisterx;
using namespace twisterx::python::table;
using namespace twisterx::io::config;
using namespace twisterx::util::uuid;
using namespace twisterx::join::config;
using namespace arrow::py;

CxTable::CxTable(std::string id) {
  id_ = id;
}

std::string CxTable::get_id() {
  return this->id_;
}

int CxTable::columns() {
  return column_count(this->get_id());
}

int CxTable::rows() {
  return row_count(this->get_id());
}

void CxTable::clear() {

}

void CxTable::show() {
  print(this->get_id(), 0, this->columns(), 0, this->rows());
}

void CxTable::show(int row1, int row2, int col1, int col2) {
  print(this->get_id(), col1, col2, row1, row2);
}

Status CxTable::from_csv(const std::string &path, const char &delimiter, const std::string &uuid) {
  twisterx::Status status = read_csv(path, uuid, CSVReadOptions().WithDelimiter(delimiter));
  return status;
}

Status CxTable::to_csv(const std::string &path) {
  ofstream out_csv;
  out_csv.open(path);
  Status status = print_to_ostream(this->get_id(), 0,
								   this->columns(), 0,
								   this->rows(), out_csv);
  out_csv.close();
  return status;
}

std::string CxTable::from_pyarrow_table(std::shared_ptr<arrow::Table> table) {
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  put_table(uuid, table);
  return uuid;
}

std::shared_ptr<arrow::Table> CxTable::to_pyarrow_table(const std::string &table_id) {
  shared_ptr<arrow::Table> table1 = get_table(table_id);
  return table1;
}

std::string CxTable::join(const std::string &table_id,
						 JoinType type,
						 JoinAlgorithm algorithm,
						 int left_column_index,
						 int right_column_index) {
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();

  JoinConfig jc(type, left_column_index, right_column_index, algorithm);
  twisterx::Status status = twisterx::JoinTables(
	  this->get_id(),
	  table_id,
	  jc,
	  uuid
  );
  if (status.is_ok()) {
	return uuid;
  } else {
	return "";
  }
}

std::string CxTable::join(const std::string &table_id, JoinConfig join_config) {
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  twisterx::Status status = twisterx::JoinTables(
	  this->get_id(),
	  table_id,
	  join_config,
	  uuid
  );
  if (status.is_ok()) {
	return uuid;
  } else {
	return "";
  }
}
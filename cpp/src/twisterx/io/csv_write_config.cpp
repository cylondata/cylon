#include "csv_write_config.h"

namespace twisterx {
namespace io {
namespace config {

CSVWriteOptions CSVWriteOptions::WithDelimiter(char delimiter) {
  this->delimiter = delimiter;
  return *this;
}
CSVWriteOptions CSVWriteOptions::ColumnNames(const std::vector<std::string> &column_names) {
  this->column_names = column_names;
  this->override_column_names = true;
  return *this;
}
char CSVWriteOptions::GetDelimiter() const {
  return delimiter;
}
std::vector<std::string> &CSVWriteOptions::GetColumnNames() const {
  return column_names;
}
bool CSVWriteOptions::IsOverrideColumnNames() const {
  return override_column_names;
}
}
}
}
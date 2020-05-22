#ifndef TWISTERX_SRC_TWISTERX_IO_CSV_READ_CONFIG_H_
#define TWISTERX_SRC_TWISTERX_IO_CSV_READ_CONFIG_H_
#include <vector>
#include <string>

namespace twisterx {
namespace io {
namespace config {
class CSVWriteOptions {

 private:
  char delimiter = ',';
  std::vector<std::string> &column_names;
  bool override_column_names = false;

 public:
  /**
   * Change the default delimiter(",")
   * @param delimiter character representing the delimiter
   */
  CSVWriteOptions WithDelimiter(char delimiter);

  CSVWriteOptions ColumnNames(const std::vector<std::string> &column_names);

  char GetDelimiter() const;
  std::vector<std::string> &GetColumnNames() const;
  bool IsOverrideColumnNames() const;
};
}
}
}
#endif //TWISTERX_SRC_TWISTERX_IO_CSV_READ_CONFIG_H_

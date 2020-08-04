/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CYLON_SRC_CYLON_IO_CSV_WRITE_CONFIG_HPP_
#define CYLON_SRC_CYLON_IO_CSV_WRITE_CONFIG_HPP_
#include <vector>
#include <string>

namespace cylon {
namespace io {
namespace config {

class CSVWriteOptions {
public:
  CSVWriteOptions();
  /**
   * Change the default delimiter(",")
   * @param delimiter character representing the delimiter
   */
  CSVWriteOptions WithDelimiter(char delimiter);

  CSVWriteOptions ColumnNames(const std::vector<std::string> &column_names);

  char GetDelimiter() const;
  std::vector<std::string> GetColumnNames() const;
  bool IsOverrideColumnNames() const;
private:
  char delimiter = ',';
  std::vector<std::string> column_names{};
  bool override_column_names = false;
};

}  // namespace config
}  // namespace io
}  // namespace cylon

#endif //CYLON_SRC_CYLON_IO_CSV_WRITE_CONFIG_HPP_

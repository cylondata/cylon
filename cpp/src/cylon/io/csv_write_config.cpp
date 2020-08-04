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

#include "csv_write_config.hpp"

namespace cylon {
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

std::vector<std::string> CSVWriteOptions::GetColumnNames() const {
  return column_names;
}

bool CSVWriteOptions::IsOverrideColumnNames() const {
  return override_column_names;
}

CSVWriteOptions::CSVWriteOptions() {
}

}  // namespace config
}  // namespace io
}  // namespace cylon

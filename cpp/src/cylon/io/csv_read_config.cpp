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

#include <utility>

#include "csv_read_config.hpp"
#include "csv_read_config_holder.hpp"
#include "../arrow//arrow_types.hpp"

namespace cylon {
namespace io {
namespace config {

CSVReadOptions CSVReadOptions::UseThreads(bool use_threads) {
  CSVConfigHolder::GetCastedHolder(*this)->use_threads = use_threads;
  return *this;
}
CSVReadOptions CSVReadOptions::WithDelimiter(char delimiter) {
  CSVConfigHolder::GetCastedHolder(*this)->delimiter = delimiter;
  return *this;
}
CSVReadOptions CSVReadOptions::IgnoreEmptyLines() {
  CSVConfigHolder::GetCastedHolder(*this)->ignore_empty_lines = true;
  return *this;
}
CSVReadOptions CSVReadOptions::AutoGenerateColumnNames() {
  CSVConfigHolder::GetCastedHolder(*this)->autogenerate_column_names = true;
  return *this;
}
CSVReadOptions CSVReadOptions::ColumnNames(const std::vector<std::string> &column_names) {
  CSVConfigHolder::GetCastedHolder(*this)->column_names = column_names;
  return *this;
}
CSVReadOptions CSVReadOptions::BlockSize(int block_size) {
  CSVConfigHolder::GetCastedHolder(*this)->block_size = block_size;
  return *this;
}
CSVReadOptions CSVReadOptions::SkipRows(int32_t skip_rows) {
  CSVConfigHolder::GetCastedHolder(*this)->skip_rows = skip_rows;
  return *this;
}

CSVReadOptions::CSVReadOptions() {
  CSVReadOptions::holder = std::shared_ptr<void>(new CSVConfigHolder());
}

std::shared_ptr<void> CSVReadOptions::GetHolder() const {
  return CSVReadOptions::holder;
}
CSVReadOptions CSVReadOptions::UseQuoting() {
  CSVConfigHolder::GetCastedHolder(*this)->quoting = true;
  return *this;
}
CSVReadOptions CSVReadOptions::WithQuoteChar(char quote_char) {
  CSVConfigHolder::GetCastedHolder(*this)->quote_char = quote_char;
  return *this;
}
CSVReadOptions CSVReadOptions::DoubleQuote() {
  CSVConfigHolder::GetCastedHolder(*this)->double_quote = true;
  return *this;
}
CSVReadOptions CSVReadOptions::UseEscaping() {
  CSVConfigHolder::GetCastedHolder(*this)->escaping = true;
  return *this;
}
CSVReadOptions CSVReadOptions::EscapingCharacter(char escaping_char) {
  CSVConfigHolder::GetCastedHolder(*this)->escape_char = escaping_char;
  return *this;
}
CSVReadOptions CSVReadOptions::HasNewLinesInValues() {
  CSVConfigHolder::GetCastedHolder(*this)->newlines_in_values = true;
  return *this;
}
CSVReadOptions CSVReadOptions::WithColumnTypes(const std::unordered_map<std::string,
                                               std::shared_ptr<DataType>> &column_types) {
  std::unordered_map<std::string,
                     std::shared_ptr<arrow::DataType>> arrow_types{};

  for (const auto &column_type : column_types) {
    auto pr = std::pair<std::string, std::shared_ptr<arrow::DataType>>(column_type.first,
                                cylon::tarrow::convertToArrowType(column_type.second));
    arrow_types.insert(pr);
  }
  CSVConfigHolder::GetCastedHolder(*this)->column_types = arrow_types;
  return *this;
}
CSVReadOptions CSVReadOptions::NullValues(const std::vector<std::string> &null_value) {
  CSVConfigHolder::GetCastedHolder(*this)->null_values = null_value;
  return *this;
}
CSVReadOptions CSVReadOptions::TrueValues(const std::vector<std::string> &true_values) {
  CSVConfigHolder::GetCastedHolder(*this)->true_values = true_values;
  return *this;
}
CSVReadOptions CSVReadOptions::FalseValues(const std::vector<std::string> &false_values) {
  CSVConfigHolder::GetCastedHolder(*this)->false_values = false_values;
  return *this;
}
CSVReadOptions CSVReadOptions::StringsCanBeNull() {
  CSVConfigHolder::GetCastedHolder(*this)->strings_can_be_null = true;
  return *this;
}
CSVReadOptions CSVReadOptions::IncludeColumns(const std::vector<std::string> &include_columns) {
  CSVConfigHolder::GetCastedHolder(*this)->include_columns = include_columns;
  return *this;
}
CSVReadOptions CSVReadOptions::IncludeMissingColumns() {
  CSVConfigHolder::GetCastedHolder(*this)->include_missing_columns = true;
  return *this;
}
CSVReadOptions CSVReadOptions::ConcurrentFileReads(bool concurrent_file_reads) {
  this->concurrent_file_reads = concurrent_file_reads;
  return *this;
}
bool CSVReadOptions::IsConcurrentFileReads() {
  return this->concurrent_file_reads;
}
}  // namespace config
}  // namespace io
}  // namespace cylon

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

#include <memory>

#include "row.hpp"
#include "table_api_extended.hpp"

namespace cylon {

template<typename ARROW_TYPE>
auto get_numeric(const std::string &table_id, int64_t col_index, int64_t row_index) {
  std::shared_ptr<arrow::Table> table = GetTable(table_id);
  auto numeric_array = std::static_pointer_cast<arrow::NumericArray<ARROW_TYPE>>(
      table->column(col_index)->chunk(0));
  return numeric_array->Value(row_index);
}

Row:: Row(const std::string &table_id, int64_t row_index) {
  this->table_id = table_id;
  this->row_index = row_index;
}

int8_t Row::GetInt8(int64_t col_index) {
  return get_numeric<arrow::Int8Type>(this->table_id, col_index, this->row_index);
}

uint8_t Row::GetUInt8(int64_t col_index) {
  return get_numeric<arrow::UInt8Type>(this->table_id, col_index, this->row_index);
}

int16_t Row::GetInt16(int64_t col_index) {
  return get_numeric<arrow::Int16Type>(this->table_id, col_index, this->row_index);
}

uint16_t Row::GetUInt16(int64_t col_index) {
  return get_numeric<arrow::UInt16Type>(this->table_id, col_index, this->row_index);
}

int32_t Row::GetInt32(int64_t col_index) {
  return get_numeric<arrow::Int32Type>(this->table_id, col_index, this->row_index);
}

uint32_t Row::GetUInt32(int64_t col_index) {
  return get_numeric<arrow::UInt32Type>(this->table_id, col_index, this->row_index);
}

int64_t Row::GetInt64(int64_t col_index) {
  return get_numeric<arrow::Int64Type>(this->table_id, col_index, this->row_index);
}

uint64_t Row::GetUInt64(int64_t col_index) {
  return get_numeric<arrow::UInt64Type>(this->table_id, col_index, this->row_index);
}

int64_t Row::RowIndex() {
  return this->row_index;
}

bool Row::GetBool(int64_t col_index) {
  std::shared_ptr<arrow::Table> table = GetTable(table_id);
  auto numeric_array = std::static_pointer_cast<arrow::BooleanArray>(
      table->column(col_index)->chunk(0));
  return numeric_array->Value(row_index);
}

float Row::GetHalfFloat(int64_t col_index) {
  return get_numeric<arrow::HalfFloatType>(this->table_id, col_index, this->row_index);
}

float Row::GetFloat(int64_t col_index) {
  return get_numeric<arrow::FloatType>(this->table_id, col_index, this->row_index);
}

double Row::GetDouble(int64_t col_index) {
  return get_numeric<arrow::DoubleType>(this->table_id, col_index, this->row_index);
}

std::string Row::GetString(int64_t col_index) {
  std::shared_ptr<arrow::Table> table = GetTable(table_id);
  auto numeric_array = std::static_pointer_cast<arrow::StringArray>(
      table->column(col_index)->chunk(0));
  return numeric_array->GetString(row_index);
}

const uint8_t *Row::GetFixedBinary(int64_t col_index) {
  std::shared_ptr<arrow::Table> table = GetTable(table_id);
  auto numeric_array = std::static_pointer_cast<arrow::FixedSizeBinaryArray>(
      table->column(col_index)->chunk(0));
  return numeric_array->GetValue(row_index);
}

int32_t Row::GetDate32(int64_t col_index) {
  return get_numeric<arrow::Date32Type>(this->table_id, col_index, this->row_index);
}

int64_t Row::GetDate64(int64_t col_index) {
  return get_numeric<arrow::Date64Type>(this->table_id, col_index, this->row_index);
}
int64_t Row::GetTimestamp(int64_t col_index) {
  return get_numeric<arrow::TimestampType>(this->table_id, col_index, this->row_index);
}

int32_t Row::Time32(int64_t col_index) {
  return get_numeric<arrow::Time32Type>(this->table_id, col_index, this->row_index);
}

int64_t Row::Time64(int64_t col_index) {
  return get_numeric<arrow::Time64Type>(this->table_id, col_index, this->row_index);
}

const uint8_t * Row::Decimal(int64_t col_index) {
  std::shared_ptr<arrow::Table> table = GetTable(table_id);
  auto numeric_array = std::static_pointer_cast<arrow::DecimalArray>(
      table->column(col_index)->chunk(0));
  return numeric_array->GetValue(row_index);
}

}  // namespace cylon

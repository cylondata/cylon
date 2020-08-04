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

#include "arrow_builder.hpp"
#include <glog/logging.h>
#include <arrow/ipc/reader.h>
#include <iostream>
#include <utility>
#include <unordered_map>
#include <memory>

#include "../table_api_extended.hpp"

std::unordered_map<std::string,
std::shared_ptr<std::vector < std::shared_ptr < arrow::Array>>>>
columns;
std::unordered_map<std::string, std::shared_ptr<std::vector < std::shared_ptr < arrow::Field>>>>
fields;

cylon::Status cylon::cyarrow::BeginTable(const std::string &table_id) {
  auto columns_vector = std::make_shared < std::vector < std::shared_ptr < arrow::Array>>>();
  auto fields_vector = std::make_shared < std::vector < std::shared_ptr < arrow::Field>>>();
  columns.insert(std::make_pair(table_id, columns_vector));
  fields.insert(std::make_pair(table_id, fields_vector));
  return cylon::Status::OK();
}

std::shared_ptr<arrow::DataType> GetArrowType(int8_t type) {
  switch (type) {
    case arrow::Type::NA:return arrow::null();
    case arrow::Type::BOOL:return arrow::boolean();
    case arrow::Type::UINT8:return arrow::uint8();
    case arrow::Type::INT8:return arrow::int8();
    case arrow::Type::UINT16:return arrow::uint16();
    case arrow::Type::INT16:return arrow::int16();
    case arrow::Type::UINT32:return arrow::uint32();
    case arrow::Type::INT32:return arrow::int32();
    case arrow::Type::UINT64:return arrow::uint64();
    case arrow::Type::INT64:return arrow::int64();
    case arrow::Type::HALF_FLOAT:return arrow::float16();
    case arrow::Type::FLOAT:return arrow::float32();
    case arrow::Type::DOUBLE:return arrow::float64();
    case arrow::Type::STRING:return arrow::utf8();
    case arrow::Type::BINARY:return arrow::binary();
    case arrow::Type::FIXED_SIZE_BINARY:break;
    case arrow::Type::DATE32:break;
    case arrow::Type::DATE64:break;
    case arrow::Type::TIMESTAMP:break;
    case arrow::Type::TIME32:break;
    case arrow::Type::TIME64:break;
    case arrow::Type::INTERVAL:break;
    case arrow::Type::DECIMAL:break;
    case arrow::Type::LIST:break;
    case arrow::Type::STRUCT:break;
    case arrow::Type::UNION:break;
    case arrow::Type::DICTIONARY:break;
    case arrow::Type::MAP:break;
    case arrow::Type::EXTENSION:break;
    case arrow::Type::FIXED_SIZE_LIST:break;
    case arrow::Type::DURATION:break;
    case arrow::Type::LARGE_STRING:break;
    case arrow::Type::LARGE_BINARY:break;
    case arrow::Type::LARGE_LIST:break;
  }
  return nullptr;
}

void AddColumnToTable(const std::string &table_id,
                      const std::string &col_name,
                      int32_t values_count,
                      int32_t null_count,
                      const std::vector<std::shared_ptr<arrow::Buffer>> &buffers,
                      const std::shared_ptr<arrow::DataType> &data_type) {
  LOG(INFO) << "Adding column of type " << data_type->name() << " to the table";
  auto array_data = arrow::ArrayData::Make(
      data_type,
      values_count,
      buffers,
      null_count);

  LOG(INFO) << "length : " << array_data->length << ", expected size : " << values_count;

  columns.find(table_id)->second->push_back(arrow::MakeArray(array_data));
  fields.find(table_id)->second->push_back(arrow::field(col_name, data_type));
}

cylon::Status cylon::cyarrow::AddColumn(const std::string &table_id,
                                        const std::string &col_name,
                                        int8_t type,
                                        int32_t value_count,
                                        int32_t null_count,
                                        int64_t validity_address, int64_t validity_size,
                                        int64_t data_address, int64_t data_size) {
  auto validity_buff = arrow::Buffer::Wrap(reinterpret_cast<uint8_t *>(validity_address),
                                           validity_size);
  auto data_buff = arrow::Buffer::Wrap(reinterpret_cast<uint8_t *>(data_address), data_size);

  auto arrow_type = GetArrowType(type);
  LOG(INFO) << "Preparing to add column of type " << arrow_type->name() << " to tale "
            << table_id << ", column " << col_name;
  AddColumnToTable(table_id, col_name, value_count, null_count,
                   {validity_buff, data_buff}, arrow_type);
  return cylon::Status::OK();
}

cylon::Status cylon::cyarrow::FinishTable(const std::string &table_id) {
  // building schema
  arrow::SchemaBuilder schema_builder;
  auto status = schema_builder.AddFields(*fields.find(table_id)->second);

  if (!status.ok()) {
    return cylon::Status(static_cast<int>(status.code()), status.message());
  }

  auto schema_result = schema_builder.Finish();
  if (!schema_result.ok()) {
    return cylon::Status(static_cast<int>(schema_result.status().code()),
        schema_result.status().message());
  }

  // building the table
  auto table = arrow::Table::Make(schema_result.ValueOrDie(), *columns.find(table_id)->second);
  cylon::PutTable(table_id, table);
  return cylon::Status::OK();
}

cylon::Status cylon::cyarrow::AddColumn(const std::string &table_id,
                                        const std::string &col_name,
                                        int8_t type,
                                        int32_t value_count,
                                        int32_t null_count,
                                        int64_t validity_address,
                                        int64_t validity_size,
                                        int64_t data_address,
                                        int64_t data_size,
                                        int64_t offset_address,
                                        int64_t offset_size) {
  LOG(INFO) <<"offset size "<< offset_size;
  auto validity_buff = arrow::Buffer::Wrap(reinterpret_cast<uint8_t *>(validity_address), validity_size);
  auto data_buff = arrow::Buffer::Wrap(reinterpret_cast<uint8_t *>(data_address), data_size);
  auto offset_buff = arrow::Buffer::Wrap(reinterpret_cast<uint8_t *>(offset_address), offset_size);

  auto arrow_type = GetArrowType(type);
  LOG(INFO) << "Preparing to add column of type " << arrow_type->name() << " to tale "
            << table_id << ", column " << col_name;
  AddColumnToTable(table_id, col_name, value_count, null_count,
                   {validity_buff, offset_buff, data_buff}, arrow_type);
  return cylon::Status::OK();
}

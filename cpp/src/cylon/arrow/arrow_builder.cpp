#include "arrow_builder.hpp"
#include "../table_api_extended.hpp"
#include <arrow/ipc/reader.h>
#include <iostream>
#include <utility>
#include <glog/logging.h>

std::unordered_map<std::string,
                   std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>> columns;
std::unordered_map<std::string, std::shared_ptr<std::vector<std::shared_ptr<arrow::Field>>>> fields;

cylon::Status cylon::cyarrow::BeginTable(const std::string &table_id) {
  auto columns_vector = std::make_shared<std::vector<std::shared_ptr<arrow::Array>>>();
  auto fields_vector = std::make_shared<std::vector<std::shared_ptr<arrow::Field>>>();
  columns.insert(std::make_pair(table_id, columns_vector));
  fields.insert(std::make_pair(table_id, fields_vector));
  return cylon::Status::OK();
}

template<typename TYPE>
void AddColumnToTable(const std::string &table_id,
                      const std::string &col_name,
                      std::shared_ptr<arrow::Buffer> buffer, int64_t size) {
  auto array_data = std::make_shared<arrow::ArrayData>(std::shared_ptr<TYPE>(),
                                                       size,
                                                       std::vector<std::shared_ptr<arrow::Buffer>>{std::move(buffer)});
  columns.find(table_id)->second->push_back(arrow::MakeArray(array_data));
  fields.find(table_id)->second->push_back(arrow::field(col_name, std::shared_ptr<TYPE>()));
}

cylon::Status cylon::cyarrow::AddColumn(const std::string &table_id,
                                        const std::string &col_name,
                                        int32_t type,
                                        int64_t address,
                                        int64_t size) {
  auto *buffer = reinterpret_cast<uint8_t *>(address);
  auto buff = std::make_shared<arrow::Buffer>(buffer, size);
  switch (type) {
    case arrow::Type::NA:break;
    case arrow::Type::BOOL:AddColumnToTable<arrow::BooleanType>(table_id, col_name, buff, size);
    case arrow::Type::UINT8:break;
    case arrow::Type::INT8:break;
    case arrow::Type::UINT16:break;
    case arrow::Type::INT16:break;
    case arrow::Type::UINT32:break;
    case arrow::Type::INT32:AddColumnToTable<arrow::Int32Type>(table_id, col_name, buff, size);
      break;
    case arrow::Type::UINT64:break;
    case arrow::Type::INT64:break;
    case arrow::Type::HALF_FLOAT:break;
    case arrow::Type::FLOAT:break;
    case arrow::Type::DOUBLE:AddColumnToTable<arrow::DoubleType>(table_id, col_name, buff, size);
      break;
    case arrow::Type::STRING:break;
    case arrow::Type::BINARY:break;
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
  return cylon::Status::OK();
}
cylon::Status cylon::cyarrow::EndTable(const std::string &table_id) {
  // building schema
  arrow::SchemaBuilder schema_builder;
  auto status = schema_builder.AddFields(*fields.find(table_id)->second);

  if (!status.ok()) {
    return cylon::Status((int) status.code(), status.message());
  }

  auto schema_result = schema_builder.Finish();
  if (!schema_result.ok()) {
    return cylon::Status((int) schema_result.status().code(), schema_result.status().message());
  }

  // building the table
  auto table = arrow::Table::Make(schema_result.ValueOrDie(), *columns.find(table_id)->second);
  cylon::PutTable(table_id, table);
  return cylon::Status::OK();
}

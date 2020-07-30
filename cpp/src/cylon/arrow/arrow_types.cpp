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

#include "arrow_types.hpp"
#include "../data_types.hpp"

namespace cylon {
namespace tarrow {

std::shared_ptr <arrow::DataType> convertToArrowType(std::shared_ptr <DataType> tType,
                                                     int32_t width,
                                                     int32_t precision,
                                                     int32_t scale) {
  switch (tType->getType()) {
    case Type::BOOL:
      return std::make_shared<arrow::BooleanType>();
    case Type::UINT8:
      return std::make_shared<arrow::UInt8Type>();
    case Type::INT8:
      return std::make_shared<arrow::Int8Type>();
    case Type::UINT16:
      return std::make_shared<arrow::UInt16Type>();
    case Type::INT16:
      return std::make_shared<arrow::Int16Type>();
    case Type::UINT32:
      return std::make_shared<arrow::UInt32Type>();
    case Type::INT32:
      return std::make_shared<arrow::Int32Type>();
    case Type::UINT64:
      return std::make_shared<arrow::UInt64Type>();
    case Type::INT64:
      return std::make_shared<arrow::Int64Type>();
    case Type::HALF_FLOAT:
      return std::make_shared<arrow::HalfFloatType>();
    case Type::FLOAT:
      return std::make_shared<arrow::FloatType>();
    case Type::DOUBLE:
      return std::make_shared<arrow::DoubleType>();
    case Type::STRING:
      return std::make_shared<arrow::StringType>();
    case Type::BINARY:
      return std::make_shared<arrow::BinaryType>();
    case Type::FIXED_SIZE_BINARY:
      return std::make_shared<arrow::FixedSizeBinaryType>(width);
    case Type::DATE32:
      return std::make_shared<arrow::Date32Type>();
    case Type::DATE64:
      return std::make_shared<arrow::Date64Type>();
    case Type::TIMESTAMP:
      return std::make_shared<arrow::TimestampType>();
    case Type::TIME32:
      return std::make_shared<arrow::Time32Type>();
    case Type::TIME64:
      return std::make_shared<arrow::Time64Type>();
    case Type::DECIMAL:
      return std::make_shared<arrow::DecimalType>(width, precision, scale);
    case Type::DURATION:
      return std::make_shared<arrow::DurationType>();
    case Type::INTERVAL:
    case Type::LIST:
    case Type::FIXED_SIZE_LIST:
    case Type::EXTENSION:
      break;
  }
  return nullptr;
}

bool validateArrowTableTypes(const std::shared_ptr <arrow::Table> &table) {
  std::shared_ptr <arrow::Schema> schema = table->schema();
  for (const auto &t : schema->fields()) {
    switch (t->type()->id()) {
      case arrow::Type::NA:
        break;
      case arrow::Type::BOOL:
        break;
      case arrow::Type::UINT8:
      case arrow::Type::INT8:
      case arrow::Type::UINT16:
      case arrow::Type::INT16:
      case arrow::Type::UINT32:
      case arrow::Type::INT32:
      case arrow::Type::UINT64:
      case arrow::Type::INT64:
      case arrow::Type::HALF_FLOAT:
      case arrow::Type::FLOAT:
      case arrow::Type::DOUBLE:
      case arrow::Type::BINARY:
      case arrow::Type::FIXED_SIZE_BINARY:
        return true;
      case arrow::Type::STRING:
        break;
      case arrow::Type::DATE32:
        break;
      case arrow::Type::DATE64:
        break;
      case arrow::Type::TIMESTAMP:
        break;
      case arrow::Type::TIME32:
        break;
      case arrow::Type::TIME64:
        break;
      case arrow::Type::INTERVAL:
        break;
      case arrow::Type::DECIMAL:
        break;
      case arrow::Type::LIST: {
        auto t_value = std::static_pointer_cast<arrow::ListType>(t->type());
        switch (t_value->value_type()->id()) {
          case arrow::Type::UINT8:
          case arrow::Type::INT8:
          case arrow::Type::UINT16:
          case arrow::Type::INT16:
          case arrow::Type::UINT32:
          case arrow::Type::INT32:
          case arrow::Type::UINT64:
          case arrow::Type::INT64:
          case arrow::Type::HALF_FLOAT:
          case arrow::Type::FLOAT:
          case arrow::Type::DOUBLE:
            return true;
          default:
            return false;
        }
      }
      case arrow::Type::STRUCT:
      case arrow::Type::UNION:
      case arrow::Type::DICTIONARY:
      case arrow::Type::MAP:
      case arrow::Type::EXTENSION:
      case arrow::Type::FIXED_SIZE_LIST:
      case arrow::Type::DURATION:
      case arrow::Type::LARGE_STRING:
      case arrow::Type::LARGE_BINARY:
      case arrow::Type::LARGE_LIST:
        return false;
    }
  }
  return false;
}

}  // namespace tarrow
}  // namespace cylon

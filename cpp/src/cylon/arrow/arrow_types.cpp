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
#include "glog/logging.h"

namespace cylon {
namespace tarrow {

std::shared_ptr<arrow::DataType> convertToArrowType(const std::shared_ptr<DataType> &tType,
                                                    int32_t width,
                                                    int32_t precision,
                                                    int32_t scale) {
  switch (tType->getType()) {
    case Type::BOOL:return std::make_shared<arrow::BooleanType>();
    case Type::UINT8:return std::make_shared<arrow::UInt8Type>();
    case Type::INT8:return std::make_shared<arrow::Int8Type>();
    case Type::UINT16:return std::make_shared<arrow::UInt16Type>();
    case Type::INT16:return std::make_shared<arrow::Int16Type>();
    case Type::UINT32:return std::make_shared<arrow::UInt32Type>();
    case Type::INT32:return std::make_shared<arrow::Int32Type>();
    case Type::UINT64:return std::make_shared<arrow::UInt64Type>();
    case Type::INT64:return std::make_shared<arrow::Int64Type>();
    case Type::HALF_FLOAT:return std::make_shared<arrow::HalfFloatType>();
    case Type::FLOAT:return std::make_shared<arrow::FloatType>();
    case Type::DOUBLE:return std::make_shared<arrow::DoubleType>();
    case Type::STRING:return std::make_shared<arrow::StringType>();
    case Type::BINARY:return std::make_shared<arrow::BinaryType>();
    case Type::FIXED_SIZE_BINARY: {
      if (width < 0) break;
      return std::make_shared<arrow::FixedSizeBinaryType>(width);
    }
    case Type::DATE32:return std::make_shared<arrow::Date32Type>();
    case Type::DATE64:return std::make_shared<arrow::Date64Type>();
    case Type::TIMESTAMP:return std::make_shared<arrow::TimestampType>();
    case Type::TIME32:return std::make_shared<arrow::Time32Type>();
    case Type::TIME64:return std::make_shared<arrow::Time64Type>();
    case Type::DECIMAL: {
      if (width < 0 || precision < 0 || scale < 0) break;
      return std::make_shared<arrow::DecimalType>(width, precision, scale);
    }
    case Type::DURATION:return std::make_shared<arrow::DurationType>();
    case Type::INTERVAL:break;
    case Type::LIST:break;
    case Type::FIXED_SIZE_LIST:break;
    case Type::EXTENSION:break;
  }
  return nullptr;
}

bool validateArrowTableTypes(const std::shared_ptr <arrow::Table> &table) {
  std::shared_ptr <arrow::Schema> schema = table->schema();
  
  for (const auto &t : schema->fields()) {
    switch (t->type()->id()) {
      case arrow::Type::BOOL:continue;
      case arrow::Type::UINT8:continue;
      case arrow::Type::INT8:continue;
      case arrow::Type::UINT16:continue;
      case arrow::Type::INT16:continue;
      case arrow::Type::UINT32:continue;
      case arrow::Type::INT32:continue;
      case arrow::Type::UINT64:continue;
      case arrow::Type::INT64:continue;
      case arrow::Type::HALF_FLOAT:continue;
      case arrow::Type::FLOAT:continue;
      case arrow::Type::DOUBLE:continue;
      case arrow::Type::BINARY:continue;
      case arrow::Type::FIXED_SIZE_BINARY:continue;
      case arrow::Type::STRING: continue; // types above are allowed. go to next column type
      case arrow::Type::LIST: {
        auto t_value = std::static_pointer_cast<arrow::ListType>(t->type());
        switch (t_value->value_type()->id()) {
          case arrow::Type::UINT8:continue;
          case arrow::Type::INT8:continue;
          case arrow::Type::UINT16:continue;
          case arrow::Type::INT16:continue;
          case arrow::Type::UINT32:continue;
          case arrow::Type::INT32:continue;
          case arrow::Type::UINT64:continue;
          case arrow::Type::INT64:continue;
          case arrow::Type::HALF_FLOAT:continue;
          case arrow::Type::FLOAT:continue;
          case arrow::Type::DOUBLE:continue; // types above are allowed. go to next column type
          default:return false;
        }
      }
      case arrow::Type::NA:return false;
      case arrow::Type::DATE32:return true;
      case arrow::Type::DATE64:return true;
      case arrow::Type::TIMESTAMP:return true;
      case arrow::Type::TIME32:return true;
      case arrow::Type::TIME64:return true;
      case arrow::Type::DECIMAL:return false;
      case arrow::Type::STRUCT:return false;
      case arrow::Type::DICTIONARY:return false;
      case arrow::Type::MAP:return false;
      case arrow::Type::EXTENSION:return false;
      case arrow::Type::FIXED_SIZE_LIST:return false;
      case arrow::Type::DURATION:return false;
      case arrow::Type::LARGE_STRING:return false;
      case arrow::Type::LARGE_BINARY:return false;
      case arrow::Type::LARGE_LIST:return false;
      case arrow::Type::INTERVAL_MONTHS:return false;
      case arrow::Type::INTERVAL_DAY_TIME:return false;
      case arrow::Type::SPARSE_UNION:return false;
      case arrow::Type::DENSE_UNION:return false;
      case arrow::Type::MAX_ID:return false; // types above are NOT allowed. return false
    }
  }
  return true;
}

std::shared_ptr<DataType> ToCylonType(const std::shared_ptr<arrow::DataType> &arr_type) {
  switch (arr_type->id()) {
    case arrow::Type::BOOL:return cylon::Bool();
    case arrow::Type::UINT8:return cylon::UInt8();
    case arrow::Type::INT8:return cylon::Int8();
    case arrow::Type::UINT16:return cylon::UInt16();
    case arrow::Type::INT16:return cylon::Int16();
    case arrow::Type::UINT32:return cylon::UInt32();
    case arrow::Type::INT32:return cylon::Int32();
    case arrow::Type::UINT64:return cylon::UInt64();
    case arrow::Type::INT64:return cylon::Int64();
    case arrow::Type::HALF_FLOAT:return cylon::HalfFloat();
    case arrow::Type::FLOAT:return cylon::Float();
    case arrow::Type::DOUBLE:return cylon::Double();
    case arrow::Type::BINARY:return cylon::Binary();
    case arrow::Type::FIXED_SIZE_BINARY:return cylon::FixedBinary();
    case arrow::Type::STRING:return cylon::String();
    case arrow::Type::DATE32:return cylon::Date32();
    case arrow::Type::DATE64:return cylon::Date64();
    case arrow::Type::TIMESTAMP:return cylon::Timestamp();
    case arrow::Type::TIME32:return cylon::Time32();
    case arrow::Type::TIME64:return cylon::Time64();
    case arrow::Type::DECIMAL:return cylon::Decimal();
    case arrow::Type::NA:break;
    case arrow::Type::INTERVAL_MONTHS:break;
    case arrow::Type::INTERVAL_DAY_TIME:break;
    case arrow::Type::LIST:break;
    case arrow::Type::STRUCT:break;
    case arrow::Type::SPARSE_UNION:break;
    case arrow::Type::DENSE_UNION:break;
    case arrow::Type::DICTIONARY:break;
    case arrow::Type::MAP:break;
    case arrow::Type::EXTENSION:break;
    case arrow::Type::FIXED_SIZE_LIST:break;
    case arrow::Type::DURATION:break;
    case arrow::Type::LARGE_STRING:break;
    case arrow::Type::LARGE_BINARY:break;
    case arrow::Type::LARGE_LIST:break;
    case arrow::Type::MAX_ID:break;
  }
  return nullptr;
}

}  // namespace tarrow
}  // namespace cylon

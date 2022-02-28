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

#include <cylon/arrow/arrow_types.hpp>
#include <cylon/data_types.hpp>

#include <glog/logging.h>

namespace cylon {
namespace tarrow {

std::shared_ptr<arrow::DataType> ToArrowType(const std::shared_ptr<DataType> &type) {
  switch (type->getType()) {
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
    case Type::FIXED_SIZE_BINARY:
      return arrow::fixed_size_binary(std::static_pointer_cast<FixedSizeBinaryType>(type)
                                          ->byte_width_);
    case Type::DATE32:return std::make_shared<arrow::Date32Type>();
    case Type::DATE64:return std::make_shared<arrow::Date64Type>();
    case Type::TIMESTAMP: {
      const auto &casted = std::static_pointer_cast<TimestampType>(type);
      return arrow::timestamp(ToArrowTimeUnit(casted->unit_), casted->timezone_);
    }
    case Type::TIME32:return std::make_shared<arrow::Time32Type>();
    case Type::TIME64:return std::make_shared<arrow::Time64Type>();
    case Type::DURATION: {
      const auto &casted = std::static_pointer_cast<DurationType>(type);
      return std::make_shared<arrow::DurationType>(ToArrowTimeUnit(casted->unit_));
    }
    case Type::LARGE_STRING:return std::make_shared<arrow::LargeStringType>();
    case Type::LARGE_BINARY:return std::make_shared<arrow::LargeBinaryType>();
    case Type::DECIMAL: {
      const auto &casted = std::static_pointer_cast<DecimalType>(type);
      if (casted->byte_width_ == 16) return arrow::decimal128(casted->precision_, casted->scale_);
      else if (casted->byte_width_ == 32)
        return arrow::decimal256(casted->precision_, casted->scale_);
      else break;
    }
    case Type::INTERVAL:break;
    case Type::LIST:break;
    case Type::FIXED_SIZE_LIST:break;
    case Type::EXTENSION:break;
    case Type::MAX_ID:break;
  }
  return nullptr;
}

Status CheckSupportedTypes(const std::shared_ptr<arrow::Table> &table) {
  const auto &schema = table->schema();
  for (const auto &t: schema->fields()) {
    switch (t->type()->id()) {
      /* following types are supported. go to next column type */
      case arrow::Type::BOOL:
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
      case arrow::Type::FIXED_SIZE_BINARY:
      case arrow::Type::BINARY:
      case arrow::Type::STRING:
      case arrow::Type::LARGE_BINARY:
      case arrow::Type::LARGE_STRING:
      case arrow::Type::DATE32:
      case arrow::Type::DATE64:
      case arrow::Type::TIMESTAMP:
      case arrow::Type::TIME32:
      case arrow::Type::TIME64: continue;
      case arrow::Type::LIST: {
        const auto &t_value = std::static_pointer_cast<arrow::ListType>(t->type());
        switch (t_value->value_type()->id()) {
          /* following types are supported. go to next column type */
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
          case arrow::Type::DOUBLE:continue;
          default:
            return {Code::NotImplemented,
                    "unsupported value type for lists " + t_value->value_type()->ToString()};;
        }
      }
      default: return {Code::NotImplemented, "unsupported type " + t->type()->ToString()};
    }
  }
  return Status::OK();
}

TimeUnit::type ToCylonTimeUnit(arrow::TimeUnit::type a_time_unit) {
  switch (a_time_unit) {
    case arrow::TimeUnit::MICRO: return TimeUnit::MICRO;
    case arrow::TimeUnit::SECOND: return TimeUnit::SECOND;
    case arrow::TimeUnit::MILLI: return TimeUnit::MILLI;
    case arrow::TimeUnit::NANO: return TimeUnit::NANO;
  }
  return TimeUnit::MICRO;
}

arrow::TimeUnit::type ToArrowTimeUnit(TimeUnit::type time_unit) {
  switch (time_unit) {
    case TimeUnit::MICRO: return arrow::TimeUnit::MICRO;
    case TimeUnit::SECOND: return arrow::TimeUnit::SECOND;
    case TimeUnit::MILLI: return arrow::TimeUnit::MILLI;
    case TimeUnit::NANO: return arrow::TimeUnit::NANO;
  }
  return arrow::TimeUnit::MICRO;
}

std::shared_ptr<DataType> ToCylonType(const std::shared_ptr<arrow::DataType> &a_type) {
  switch (a_type->id()) {
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
    case arrow::Type::FIXED_SIZE_BINARY:
      return cylon::FixedSizeBinary(std::static_pointer_cast<arrow::FixedSizeBinaryType>(a_type)
                                        ->byte_width());
    case arrow::Type::BINARY:return cylon::Binary();
    case arrow::Type::STRING:return cylon::String();
    case arrow::Type::LARGE_STRING: return cylon::LargeString();
    case arrow::Type::LARGE_BINARY: return cylon::LargeBinary();
    case arrow::Type::DATE32:return cylon::Date32();
    case arrow::Type::DATE64:return cylon::Date64();
    case arrow::Type::TIMESTAMP: {
      const auto &casted = std::static_pointer_cast<arrow::TimestampType>(a_type);
      return cylon::Timestamp(ToCylonTimeUnit(casted->unit()), casted->timezone());
    }
    case arrow::Type::TIME32:return cylon::Time32();
    case arrow::Type::TIME64:return cylon::Time64();
    case arrow::Type::DECIMAL128: {
      const auto &casted = std::static_pointer_cast<arrow::Decimal128Type>(a_type);
      return cylon::Decimal(16, casted->precision(), casted->scale());
    }
    case arrow::Type::DECIMAL256: {
      const auto &casted = std::static_pointer_cast<arrow::Decimal128Type>(a_type);
      return cylon::Decimal(32, casted->precision(), casted->scale());
    }
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
    case arrow::Type::LARGE_LIST:break;
    case arrow::Type::MAX_ID:break;
  }
  return nullptr;
}

Type::type ToCylonTypeId(const std::shared_ptr<arrow::DataType> &type) {
  switch (type->id()) {
    case arrow::Type::BOOL:return Type::BOOL;
    case arrow::Type::UINT8:return Type::UINT8;
    case arrow::Type::INT8:return Type::INT8;
    case arrow::Type::UINT16:return Type::UINT16;
    case arrow::Type::INT16:return Type::INT16;
    case arrow::Type::UINT32:return Type::UINT32;
    case arrow::Type::INT32:return Type::INT32;
    case arrow::Type::UINT64:return Type::UINT64;
    case arrow::Type::INT64:return Type::INT64;
    case arrow::Type::HALF_FLOAT:return Type::HALF_FLOAT;
    case arrow::Type::FLOAT:return Type::FLOAT;
    case arrow::Type::DOUBLE:return Type::DOUBLE;
    case arrow::Type::STRING:return Type::STRING;
    case arrow::Type::BINARY:return Type::BINARY;
    case arrow::Type::FIXED_SIZE_BINARY:return Type::FIXED_SIZE_BINARY;
    case arrow::Type::DATE32:return Type::DATE32;
    case arrow::Type::DATE64:return Type::DATE64;
    case arrow::Type::TIMESTAMP:return Type::TIMESTAMP;
    case arrow::Type::TIME32:return Type::TIME32;
    case arrow::Type::TIME64:return Type::TIME64;
    case arrow::Type::LARGE_STRING:return Type::LARGE_STRING;
    case arrow::Type::LARGE_BINARY:return Type::LARGE_BINARY;
    default:return Type::MAX_ID;
  }
}

}  // namespace tarrow
}  // namespace cylon

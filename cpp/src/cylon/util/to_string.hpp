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

#ifndef CYLON_SRC_UTIL_TO_STRING_H_
#define CYLON_SRC_UTIL_TO_STRING_H_

#include <string>
#include <arrow/array.h>
namespace cylon {
namespace util {

template<typename TYPE>
std::string do_to_string_numeric(const std::shared_ptr<arrow::Array> &array, int index) {
  auto casted_array = std::static_pointer_cast<arrow::NumericArray<TYPE>>(array);
  if (casted_array->IsNull(index)) {
    return "";
  }
  return std::to_string(casted_array->Value(index));
}

std::string array_to_string(const std::shared_ptr<arrow::Array> &array, int index) {
  switch (array->type()->id()) {
    case arrow::Type::NA:return "NA";
    case arrow::Type::BOOL:break;
    case arrow::Type::UINT8:return do_to_string_numeric<arrow::UInt8Type>(array, index);
    case arrow::Type::INT8:return do_to_string_numeric<arrow::Int8Type>(array, index);
    case arrow::Type::UINT16:return do_to_string_numeric<arrow::UInt16Type>(array, index);
    case arrow::Type::INT16:return do_to_string_numeric<arrow::Int8Type>(array, index);
    case arrow::Type::UINT32:return do_to_string_numeric<arrow::UInt32Type>(array, index);
    case arrow::Type::INT32:return do_to_string_numeric<arrow::Int32Type>(array, index);
    case arrow::Type::UINT64:return do_to_string_numeric<arrow::UInt64Type>(array, index);
    case arrow::Type::INT64:return do_to_string_numeric<arrow::Int64Type>(array, index);
    case arrow::Type::HALF_FLOAT:return do_to_string_numeric<arrow::HalfFloatType>(array, index);
    case arrow::Type::FLOAT:return do_to_string_numeric<arrow::FloatType>(array, index);
    case arrow::Type::DOUBLE:return do_to_string_numeric<arrow::DoubleType>(array, index);
    case arrow::Type::STRING:return std::static_pointer_cast<arrow::StringArray>(array)->GetString(index);
    case arrow::Type::BINARY:break;
    case arrow::Type::FIXED_SIZE_BINARY:break;
    case arrow::Type::DATE32:return do_to_string_numeric<arrow::Date32Type>(array, index);
    case arrow::Type::DATE64:return do_to_string_numeric<arrow::Date64Type>(array, index);
    case arrow::Type::TIMESTAMP:return do_to_string_numeric<arrow::Date64Type>(array, index);
    case arrow::Type::TIME32:return do_to_string_numeric<arrow::Time32Type>(array, index);
    case arrow::Type::TIME64:return do_to_string_numeric<arrow::Time64Type>(array, index);
    case arrow::Type::DECIMAL:break;
    case arrow::Type::LIST:break;
    case arrow::Type::STRUCT:break;
    case arrow::Type::DICTIONARY:break;
    case arrow::Type::MAP:break;
    case arrow::Type::EXTENSION:break;
    case arrow::Type::FIXED_SIZE_LIST:break;
    case arrow::Type::DURATION:break;
    case arrow::Type::LARGE_STRING:break;
    case arrow::Type::LARGE_BINARY:break;
    case arrow::Type::LARGE_LIST:break;
    case arrow::Type::INTERVAL_MONTHS:break;
    case arrow::Type::INTERVAL_DAY_TIME:break;
    case arrow::Type::SPARSE_UNION:break;
    case arrow::Type::DENSE_UNION:break;
    case arrow::Type::MAX_ID:break;
  }
  return "NA";
}
}
}

#endif //CYLON_SRC_UTIL_TO_STRING_H_

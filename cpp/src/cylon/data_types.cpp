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


#include "data_types.hpp"

namespace cylon {

std::shared_ptr<DataType> FixedSizeBinary(int32_t byte_width) {
  return std::make_shared<FixedSizeBinaryType>(byte_width);
}

std::shared_ptr<DataType> Timestamp(TimeUnit::type unit, std::string time_zone) {
  return std::make_shared<TimestampType>(unit, std::move(time_zone));
}

std::shared_ptr<DataType> Duration(TimeUnit::type unit) {
  return std::make_shared<DurationType>(unit);
}

std::shared_ptr<DataType> Decimal(int32_t byte_width, int32_t precision, int32_t scale) {
  return std::make_shared<DecimalType>(byte_width, precision, scale);
}

FixedSizeBinaryType::FixedSizeBinaryType(int32_t byte_width)
    : DataType(Type::FIXED_SIZE_BINARY, Layout::FIXED_WIDTH), byte_width_(byte_width) {}

FixedSizeBinaryType::FixedSizeBinaryType(int32_t byte_width, Type::type override_type)
    : DataType(override_type, Layout::FIXED_WIDTH), byte_width_(byte_width) {}

TimestampType::TimestampType(TimeUnit::type unit, std::string timezone)
    : DataType(Type::TIMESTAMP, Layout::FIXED_WIDTH), unit_(unit),
      timezone_(std::move(timezone)) {}

DurationType::DurationType(TimeUnit::type unit)
    : DataType(Type::DURATION, Layout::FIXED_WIDTH), unit_(unit) {}

DecimalType::DecimalType(int32_t byte_width, int32_t precision, int32_t scale)
    : FixedSizeBinaryType(byte_width, Type::DECIMAL), precision_(precision), scale_(scale) {}
}

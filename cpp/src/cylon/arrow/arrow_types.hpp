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

#ifndef CYLON_ARROW_TYPES_H
#define CYLON_ARROW_TYPES_H

#include <memory>
#include <arrow/api.h>

#include <cylon/data_types.hpp>
#include <cylon/status.hpp>

namespace cylon {
namespace tarrow {

/**
 * Convert a cylon type to an arrow type
 * @param type the cylon type
 * @return corresponding arrow type
 */
std::shared_ptr<arrow::DataType> ToArrowType(const std::shared_ptr<DataType> &type);

/**
 * Convert arrow data type pointer to Cylon Data type pointer
 * @param a_type
 * @return corresponding
 */
std::shared_ptr<DataType> ToCylonType(const std::shared_ptr<arrow::DataType> &a_type);

TimeUnit::type ToCylonTimeUnit(arrow::TimeUnit::type a_time_unit);
arrow::TimeUnit::type ToArrowTimeUnit(TimeUnit::type time_unit);

Type::type ToCylonTypeId(const std::shared_ptr<arrow::DataType> &type);

/**
 * Checks if the types of an arrow table are supported in Cylon
 * @param table true if we support the types
 * @return false if we don't support the types
 */
cylon::Status CheckSupportedTypes(const std::shared_ptr<arrow::Table> &table);

}  // namespace tarrow
}  // namespace cylon

#endif //CYLON_ARROW_TYPES_H

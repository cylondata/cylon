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

#include "../data_types.hpp"

namespace cylon {
namespace tarrow {

/**
 * Convert a cylon type to an arrow type
 * @param tType the cylon type
 * @return corresponding arrow type
 */
std::shared_ptr<arrow::DataType> convertToArrowType(std::shared_ptr<DataType> tType,
                                                    int32_t width = -1,
                                                    int32_t precision = -1,
                                                    int32_t scale = -1);

/**
 * Validate the types of an arrow table
 * @param table true if we support the types
 * @return false if we don't support the types
 */
bool validateArrowTableTypes(const std::shared_ptr<arrow::Table> &table);

}  // namespace tarrow
}  // namespace cylon

#endif //CYLON_ARROW_TYPES_H

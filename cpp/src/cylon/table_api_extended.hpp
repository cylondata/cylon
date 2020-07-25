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

#ifndef CYLON_SRC_CYLON_TABLE_API_EXTENDED_HPP_
#define CYLON_SRC_CYLON_TABLE_API_EXTENDED_HPP_

#include <arrow/api.h>

#include "status.hpp"

namespace cylon {

std::shared_ptr<arrow::Table> GetTable(const std::string &id);

void PutTable(const std::string &id, const std::shared_ptr<arrow::Table> &table);

Status VerifyTableSchema(const std::shared_ptr<arrow::Table> &ltab, const std::shared_ptr<arrow::Table> &rtab);
}  // namespace cylon
#endif //CYLON_SRC_CYLON_TABLE_API_EXTENDED_HPP_

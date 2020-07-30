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

#ifndef CYLON_SRC_IO_COLUMN_H_
#define CYLON_SRC_IO_COLUMN_H_

#include <string>
#include <utility>
#include <memory>
#include "data_types.hpp"

#include <arrow/api.h>
#include <arrow/table.h>

namespace cylon {

class Column {
 public:
  Column(std::string id, std::shared_ptr<DataType> type) : id(std::move(id)), type(std::move(type)) {
  }

  /**
   * From the arrow array
   * @param array the array to use
   * @return the column
   */
  std::shared_ptr<Column> FromArrow(std::shared_ptr<arrow::Array> array);

  /**
   * Return the unique id of the array
   * @return
   */
  std::string get_id() {
	return this->id;
  }

 private:
  /**
   * The id of the column
   */
  std::string id;

  /**
   * The datatype of the column
   */
  std::shared_ptr<DataType> type;
};

}  // namespace cylon
#endif //CYLON_SRC_IO_COLUMN_H_


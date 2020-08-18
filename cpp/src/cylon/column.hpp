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
#include "arrow/arrow_types.hpp"

#include <arrow/api.h>
#include <arrow/table.h>

namespace cylon {

class Column {
 public:
  Column(std::string id, std::shared_ptr<DataType> type)
      : id(std::move(id)), type(std::move(type)) {
  }

  /**
   * From the arrow array
   * @param array the array to use
   * @return the column
   */
  std::shared_ptr<Column> FromArrow(std::shared_ptr<arrow::Array> &array);

  std::shared_ptr<arrow::Array> GetColumnData();

  /**
   * Return the unique id of the array
   * @return
   */
  const std::string GetID() {
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

 protected:
  std::shared_ptr<arrow::Array> data;
};

#define ROUND_UP_BYTES(N) ((N + 7) / 8)

template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
class VectorColumn : public Column {
 public:
  VectorColumn(const std::string &id,
               const std::shared_ptr<DataType> &type,
               const std::vector<T> &data_vector) :
      Column(id, type),
      // define the validity array based on the size and fill it with 0xff
      validity_vector(std::vector<uint8_t>((data_vector.size() + 7) / 8, 0xff)) {

    // create the mask for the last byte
    validity_vector.back() = 1 < (8 + data_vector.size() - validity_vector.size()) - 1;

    const std::shared_ptr<arrow::Buffer> &data_buff = arrow::Buffer::Wrap(data_vector);
    const std::shared_ptr<arrow::Buffer> &val_buff = arrow::Buffer::Wrap(validity_vector);

    const std::shared_ptr<arrow::ArrayData> &arr_data =
        arrow::ArrayData::Make(cylon::tarrow::convertToArrowType(type), data_vector.size(),
                               {val_buff, data_buff});

    Column::data = arrow::MakeArray(arr_data);
  }

 private:
  std::vector<uint8_t> validity_vector;
};

}  // namespace cylon
#endif //CYLON_SRC_IO_COLUMN_H_


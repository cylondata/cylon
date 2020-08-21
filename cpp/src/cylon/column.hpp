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

#include <iostream>

namespace cylon {

class Column {
 public:
  Column(const std::string &id, const std::shared_ptr<DataType> &type,
         const std::shared_ptr<arrow::ChunkedArray> &data_)
      : id(id), type(type), data_array(data_) {
  }

  Column(const std::string &id, const std::shared_ptr<DataType> &type,
         const std::shared_ptr<arrow::Array> &data_)
      : id(id), type(type),
        data_array(std::make_shared<arrow::ChunkedArray>(data_)) {
  }

  /**
   * Return the data wrapped by column 
   * @return arrow chunked array
   */
  std::shared_ptr<arrow::ChunkedArray> GetColumnData() const;

  /**
   * Return the unique id of the array
   * @return
   */
  std::string GetID() const;

  /**
   * Return the data type of the column
   * @return
   */
  std::shared_ptr<DataType> GetDataType() const;

  static std::shared_ptr<Column> Make(const std::string &id, const std::shared_ptr<DataType> &type,
                                      const std::shared_ptr<arrow::ChunkedArray> &data_);

  static std::shared_ptr<Column> Make(const std::string &id, const std::shared_ptr<DataType> &type,
                                      const std::shared_ptr<arrow::Array> &data_);

 private:
  std::string id; // The id of the column
  std::shared_ptr<DataType> type; // The datatype of the column

 protected:
  Column(std::string id, std::shared_ptr<DataType> type)
      : id(std::move(id)), type(std::move(type)) {
  }

  std::shared_ptr<arrow::ChunkedArray> data_array;   // pointer to the data array
};

template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
class VectorColumn : public Column {
 public:
  VectorColumn(const std::string &id,
               const std::shared_ptr<DataType> &type,
               const std::shared_ptr<std::vector<T>> &data_vector) :
      Column(id, type),
      // define the validity array based on the size and fill it with 0xff
      validity_vector(std::vector<uint8_t>((data_vector->size() + 7) / 8, 0xff)) {
    // create the mask for the last byte
    validity_vector.back() = (uint8_t) (1 << (8 + data_vector->size() - 8 * validity_vector.size()))
        - 1;

    const std::shared_ptr<arrow::Buffer> &data_buff = arrow::Buffer::Wrap(*data_vector);
    const std::shared_ptr<arrow::Buffer> &val_buff = arrow::Buffer::Wrap(validity_vector);

    const std::shared_ptr<arrow::ArrayData> &arr_data =
        arrow::ArrayData::Make(cylon::tarrow::convertToArrowType(type), data_vector->size(),
                               {val_buff, data_buff});

    Column::data_array = std::make_shared<arrow::ChunkedArray>(arrow::MakeArray(arr_data));
  }

  static std::shared_ptr<VectorColumn<T>> Make(const std::string &id,
                                               const std::shared_ptr<DataType> &type,
                                               const std::shared_ptr<std::vector<T>> &data_vector) {
    return std::make_shared<VectorColumn<T>>(id, type, data_vector);
  }

 private:
  std::vector<uint8_t> validity_vector;
};

}  // namespace cylon
#endif //CYLON_SRC_IO_COLUMN_H_


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
  Column(std::string id, std::shared_ptr<DataType> type,
         std::shared_ptr<arrow::ChunkedArray> data_)
      : id(std::move(id)), type(std::move(type)), data_array(std::move(data_)) {
  }

  Column(std::string id, std::shared_ptr<DataType> type,
         const std::shared_ptr<arrow::Array> &data_)
      : id(std::move(id)), type(std::move(type)),
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
      data(data_vector) {
    const std::shared_ptr<arrow::Buffer> &data_buff = arrow::MutableBuffer::Wrap(data->data(), (int64_t)data->size());
    const std::shared_ptr<arrow::ArrayData> &arr_data =
        arrow::ArrayData::Make(cylon::tarrow::convertToArrowType(type), data->size(),
                               {nullptr, data_buff}, 0, 0);

    Column::data_array = std::make_shared<arrow::ChunkedArray>(arrow::MakeArray(arr_data));
  }

  static std::shared_ptr<VectorColumn<T>> Make(const std::string &id,
                                               const std::shared_ptr<DataType> &type,
                                               const std::shared_ptr<std::vector<T>> &data_vector) {
    return std::make_shared<VectorColumn<T>>(id, type, data_vector);
  }

 private:
  std::shared_ptr<std::vector<T>> data; // pointer to the data vector
};

}  // namespace cylon
#endif //CYLON_SRC_IO_COLUMN_H_


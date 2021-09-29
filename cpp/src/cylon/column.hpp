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
#include <arrow/api.h>
#include <arrow/table.h>

#include "cylon/data_types.hpp"
#include "cylon/ctx/arrow_memory_pool_utils.hpp"
#include "cylon/arrow/arrow_types.hpp"

namespace cylon {

class Column {
 public:
  Column(std::string id, std::shared_ptr<DataType> type,
         std::shared_ptr<arrow::ChunkedArray> data_)
      : id(std::move(id)), type(std::move(type)), data_array(std::move(data_)) {
  }

  Column(std::string id, std::shared_ptr<DataType> type,
         std::shared_ptr<arrow::Array> data_)
      : id(std::move(id)), type(std::move(type)),
        data_array(std::make_shared<arrow::ChunkedArray>(std::move(data_))) {
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

  template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
  static Status FromVector(const std::shared_ptr<CylonContext> &ctx,
                           const std::string &id,
                           const std::shared_ptr<DataType> &type,
                           const std::vector<T> &data_vector,
                           std::shared_ptr<Column> &output) {
    using ArrowT = typename arrow::CTypeTraits<T>::ArrowType;
    using BuilderT = typename arrow::TypeTraits<ArrowT>::BuilderType;

    // copy data to a buffer
    BuilderT builder(ToArrowPool(ctx));
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.AppendValues(data_vector));

    std::shared_ptr<arrow::Array> arr;
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.Finish(&arr));
    output = std::make_shared<Column>(id, type, std::move(arr));
    return Status::OK();
  }

 private:
  std::string id; // The id of the column
  std::shared_ptr<DataType> type; // The datatype of the column
  std::shared_ptr<arrow::ChunkedArray> data_array;   // pointer to the data array
};

}  // namespace cylon
#endif //CYLON_SRC_IO_COLUMN_H_


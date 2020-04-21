#ifndef TWISTERX_SRC_IO_COLUMN_H_
#define TWISTERX_SRC_IO_COLUMN_H_

#include <string>
#include <utility>
#include <memory>
#include "data_types.hpp"

#include <arrow/api.h>
#include <arrow/table.h>

namespace twisterx {

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

}
#endif //TWISTERX_SRC_IO_COLUMN_H_


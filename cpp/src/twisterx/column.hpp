#ifndef TWISTERX_SRC_IO_COLUMN_H_
#define TWISTERX_SRC_IO_COLUMN_H_

#include <string>
#include <utility>
#include <memory>
#include "data_types.h"

#include <arrow/api.h>
#include <arrow/table.h>

namespace twisterx {

class Column {
 public:
  Column(std::string  id, DataType type) : id(std::move(id)), type(type) {
  }

  /**
   * From the arrow array
   * @param array the array to use
   * @return the column
   */
  std::shared_ptr<Column> FromArrow(std::shared_ptr<arrow::Array> array);

  /**
   * Create the column from the givem buffer and data type
   * @param type the data type
   * @param buf buffer
   * @param length length of the buffer
   * @return the column
   */
  std::shared_ptr<Column> FromBuffer(DataType type, uint8_t * buf, int64_t length);

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
  DataType type;
};

}
#endif //TWISTERX_SRC_IO_COLUMN_H_


#ifndef TWISTERX_SRC_IO_COLUMN_H_
#define TWISTERX_SRC_IO_COLUMN_H_

#include <string>
#include <utility>
#include "data_types.h"

namespace twisterx {
namespace io {

class Column {
 private:
  /**
   * The id of the column
   */
  std::string id;

  /**
   * The datatype of the column
   */
  DataType type;

 public:
  Column(std::string  id, DataType type) : id(std::move(id)), type(type) {
  }

  std::string get_id() {
    return this->id;
  }
};

class ArrowColumn {

};

}
}
#endif //TWISTERX_SRC_IO_COLUMN_H_


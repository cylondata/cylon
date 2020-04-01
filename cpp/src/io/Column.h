#ifndef TWISTERX_SRC_IO_COLUMN_H_
#define TWISTERX_SRC_IO_COLUMN_H_

#include <string>
namespace twisterx {
namespace io {

template<typename TYPE>
class Column {
 private:
  std::string id;

 public:
  Column(const std::string& id) {
    this->id = id;
  }

  std::string get_id() {
    return this->id;
  }
};
}
}
#endif //TWISTERX_SRC_IO_COLUMN_H_


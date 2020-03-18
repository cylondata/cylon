#ifndef TWISTERX_SRC_IO_STATUS_H_
#define TWISTERX_SRC_IO_STATUS_H_
#include <string>
namespace twisterx {
namespace io {
class Status {
 private:
  int code;
  std::string msg;

 public:
  Status(int code, const std::string &msg) {
    this->code = code;
    this->msg = msg;
  }
  int get_code() {
    return this->code;
  }

  std::string get_msg() {
    return this->msg;
  }
};
}
}
#endif //TWISTERX_SRC_IO_STATUS_H_

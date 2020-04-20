#ifndef TWISTERX_SRC_IO_STATUS_H_
#define TWISTERX_SRC_IO_STATUS_H_
#include <string>
#include "code.cpp"

namespace twisterx {

class Status {
 private:
  int code;
  std::string msg;

 public:

  Status() {

  }

  Status(int code, const std::string &msg) {
    this->code = code;
    this->msg = msg;
  }

  explicit Status(int code) {
    this->code = code;
  }

  explicit Status(Code code_) {
    this->code = code_;
  }

  Status(Code code_, const std::string &msg) {
    this->code = code_;
    this->msg = msg;
  }

  int get_code() {
    return this->code;
  }

  bool is_ok() {
    return this->get_code() == Code::OK;
  }

  static Status OK() {
    return twisterx::Status(Code::OK);
  }

  std::string get_msg() {
    return this->msg;
  }
};
}
#endif //TWISTERX_SRC_IO_STATUS_H_

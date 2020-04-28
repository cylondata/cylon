//
// Created by vibhatha on 4/21/20.
//

#ifndef TWISTERX_STATUS_H
#define TWISTERX_STATUS_H
#include "string"
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

  explicit Status(Code code) {
    this->code = code;
  }

  Status(Code code, const std::string &msg) {
    this->code = code;
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

#endif //TWISTERX_STATUS_H

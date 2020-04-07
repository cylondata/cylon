#ifndef TWISTERX_SRC_IO_STATUS_H_
#define TWISTERX_SRC_IO_STATUS_H_
#include <string>

namespace twisterx {
enum Code {
  OK = 0,
  OutOfMemory = 1,
  KeyError = 2,
  TypeError = 3,
  Invalid = 4,
  IOError = 5,
  CapacityError = 6,
  IndexError = 7,
  UnknownError = 9,
  NotImplemented = 10,
  SerializationError = 11,
  RError = 13,
  // Gandiva range of errors
  CodeGenError = 40,
  ExpressionValidationError = 41,
  ExecutionError = 42,
  // Continue generic codes.
  AlreadyExists = 45
};

class Status {
 private:
  int code;
  std::string msg;

 public:
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
#endif //TWISTERX_SRC_IO_STATUS_H_

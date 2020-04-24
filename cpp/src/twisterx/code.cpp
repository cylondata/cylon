#ifndef TWISTERX_SRC_CODE_H_
#define TWISTERX_SRC_CODE_H_

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
}
#endif //TWISTERX_SRC_IO_STATUS_H_
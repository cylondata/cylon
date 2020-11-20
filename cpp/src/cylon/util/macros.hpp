//
// Created by niranda on 11/6/20.
//

#ifndef CYLON_CPP_SRC_CYLON_UTIL_MACROS_HPP_
#define CYLON_CPP_SRC_CYLON_UTIL_MACROS_HPP_

#define LOG_AND_RETURN_ERROR(code, msg) \
  LOG(ERROR) << msg ; \
  return cylon::Status(code, msg);

#define RETURN_IF_STATUS_FAILED(status) \
  if (!status.is_ok()) { \
    return status; \
  };


#define RETURN_IF_ARROW_STATUS_FAILED(status) \
  if (!status.ok()) { \
    return cylon::Status(static_cast<int>(status.code()), status.message()); \
  };

#endif //CYLON_CPP_SRC_CYLON_UTIL_MACROS_HPP_

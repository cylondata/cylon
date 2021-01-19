/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CYLON_CPP_SRC_CYLON_UTIL_MACROS_HPP_
#define CYLON_CPP_SRC_CYLON_UTIL_MACROS_HPP_

#define LOG_AND_RETURN_ERROR(code, msg) \
  LOG(ERROR) << msg ; \
  return cylon::Status(code, msg);

#define RETURN_CYLON_STATUS_IF_FAILED(status) \
  if (!status.is_ok()) { \
    return status; \
  };

#define LOG_AND_RETURN_CYLON_STATUS_IF_FAILED(status) \
  if (!status.is_ok()) { \
    LOG(ERROR) << status.get_msg() ; \
    return status; \
  };

#define RETURN_CYLON_STATUS_IF_ARROW_FAILED(status) \
  if (!status.ok()) { \
    return cylon::Status(static_cast<int>(status.code()), status.message()); \
  };

#define RETURN_ARROW_STATUS_IF_FAILED(status) \
  if (!status.ok()) { \
    return status; \
  };

#endif //CYLON_CPP_SRC_CYLON_UTIL_MACROS_HPP_

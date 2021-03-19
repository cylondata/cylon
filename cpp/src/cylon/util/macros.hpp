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
  return cylon::Status(code, msg)

#define RETURN_CYLON_STATUS_IF_FAILED(expr) \
  do{                                       \
    const auto& _st = (expr);               \
    if (!_st.is_ok()) {                     \
      return _st;                           \
    };                                      \
  } while (0)

#define LOG_AND_RETURN_CYLON_STATUS_IF_FAILED(expr) \
  do{                               \
    const auto& _st = (expr);       \
    if (!_st.is_ok()) {             \
      LOG(ERROR) << _st.get_msg() ; \
      return _st;                   \
    };                              \
  } while (0)

#define RETURN_CYLON_STATUS_IF_ARROW_FAILED(expr) \
  do{                               \
    const auto& _st = (expr);       \
    if (!_st.ok()) { \
      return cylon::Status(static_cast<int>(_st.code()), _st.message()); \
    };                              \
  } while (0)

#define RETURN_ARROW_STATUS_IF_FAILED(expr) \
  do{                               \
    const auto& _st = (expr);       \
    if (!_st.ok()) {                \
      return _st;                   \
    };                              \
  } while (0)

#endif //CYLON_CPP_SRC_CYLON_UTIL_MACROS_HPP_

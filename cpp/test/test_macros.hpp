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

#ifndef CYLON_CPP_TEST_TEST_MACROS_HPP_
#define CYLON_CPP_TEST_TEST_MACROS_HPP_

#include <cylon/table.hpp>
#include <cylon/table_api_extended.hpp>

#define CHECK_ARROW_EQUAL(expected, received)                                 \
  do {                                                                         \
    const auto& exp_ = (expected);                                              \
    const auto& rec_ = (received);                                              \
    INFO("Expected: " << exp_->ToString() << "\nReceived: " << rec_->ToString());\
    REQUIRE(exp_->Equals(*rec_));                                                \
  } while(0)

#define CHECK_ARROW_BUFFER_EQUAL(expected, received)                           \
  do {                                                                         \
    const auto& exp_ = (expected);                                              \
    const auto& rec_ = (received);                                              \
    INFO("Expected: " << exp_->ToHexString() << "\nReceived: " << rec_->ToHexString());\
    REQUIRE(exp_->Equals(*rec_));                                                \
  } while(0)

#define CHECK_CYLON_STATUS(expr)  \
  do{                             \
      const auto& st = (expr);    \
      INFO("code: " << st.get_code() << " msg: " << st.get_msg()); \
      REQUIRE(st.is_ok());        \
  } while(0)

#define CHECK_ARROW_STATUS(expr)  \
  do{                             \
      const auto& st = (expr);    \
      INFO("status: " << st.ToString()); \
      REQUIRE(st.ok());        \
  } while(0)

#define EXPECT_FAIL_WITH_MSG(code, msg_sub_str, expr)  \
  do{                             \
      const auto& st = (expr);                         \
      const auto& matcher = (msg_sub_str);             \
      INFO("Expected: [" << code << "-*" << matcher<< "*] \nReceived: [" \
            << st.get_code() << "-" << st.get_msg() <<"]"); \
      REQUIRE( (!st.is_ok() && st.get_msg().find(matcher) != std::string::npos));        \
  } while(0)

#define VERIFY_TABLES_EQUAL_UNORDERED(exp_expr, rec_expr)           \
  do {                                                              \
  std::shared_ptr<Table> temp;                                      \
  const auto& _expected = (exp_expr);                               \
  const auto& _result = (rec_expr);                                 \
  std::stringstream ss;                                             \
  ss << "expected:\n";                                              \
  _expected->PrintToOStream(ss);                                    \
  ss << "received:\n";                                              \
  _result->PrintToOStream(ss);                                      \
  INFO(ss.str());                                                   \
  INFO("row count: " << _expected->Rows() << " vs " << _result->Rows()); \
  REQUIRE(_expected->Rows() == _result->Rows());                    \
                                                                    \
  bool _eq_res = false;                                             \
  CHECK_CYLON_STATUS(Equals(_expected, _result, _eq_res, /*ordered=*/false)); \
  REQUIRE(_eq_res);                                                 \
} while (0)

#endif //CYLON_CPP_TEST_TEST_MACROS_HPP_

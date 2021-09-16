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

#define CHECK_ARRAYS_EQUAL(expected, received)                                 \
  do {                                                                         \
    const auto& exp = (expected);                                              \
    const auto& rec = (received);                                              \
    INFO("Expected: " << exp->ToString() << "\nReceived: " << rec->ToString()); \
    REQUIRE(exp->Equals(*rec));                                                \
  } while(0)

#define CHECK_CYLON_STATUS(expr)  \
  do{                             \
      const auto& st = (expr);    \
      INFO("code: " << st.get_code() << " msg: " << st.get_msg()); \
      REQUIRE(st.is_ok());        \
  } while(0)

#define EXPECT_FAIL_WITH_MSG(code, msg_sub_str, expr)  \
  do{                             \
      const auto& st = (expr);                         \
      const auto& matcher = (msg_sub_str);             \
      INFO("Expected: [" << code << "-*" << matcher<< "*] \nReceived: [" \
            << st.get_code() << "-" << st.get_msg() <<"]"); \
      REQUIRE( (!st.is_ok() && st.get_msg().find(matcher) != std::string::npos));        \
  } while(0)

#endif //CYLON_CPP_TEST_TEST_MACROS_HPP_

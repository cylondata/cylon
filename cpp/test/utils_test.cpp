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

#include "common/test_header.hpp"
#include "cylon/status.hpp"
#include "cylon/util/macros.hpp"
#include "test_arrow_utils.hpp"
#include "test_macros.hpp"

namespace cylon {
namespace test {

Status TestUtils() {
  CYLON_ASSIGN_OR_RAISE(auto a, MakeArrayOfNull(arrow::int32(), 4));
  REQUIRE(a->length() == 4);

  CYLON_ASSIGN_OR_RAISE(auto b, MakeArrayOfNull(arrow::int32(), 6));
  REQUIRE(b->length() == 6);
  return Status::OK();
}

TEST_CASE("Test Utils"){
  CHECK_CYLON_STATUS(TestUtils());
}

} // namespace test
} // namespace cylon
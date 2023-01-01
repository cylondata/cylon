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

TEST_CASE("Test Utils") {
  CHECK_CYLON_STATUS(TestUtils());
}

TEST_CASE("Sample test") {
  int rows = 1000;
  std::shared_ptr<Table> table;
  CHECK_CYLON_STATUS(CreateTable(ctx, rows, table));

  uint64_t count = 128;
  const auto &in_array = table->get_table()->column(0);
  std::shared_ptr<arrow::Array> out;
  CHECK_ARROW_STATUS(util::SampleArray(in_array, count, out));
  REQUIRE(in_array->type() == out->type());
  REQUIRE(out->length() == (int64_t) count);
}

TEMPLATE_LIST_TEST_CASE("Wrap numeric vector", "[utils]", ArrowNumericTypes) {
  using T = typename TestType::c_type;
  std::vector<T> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::shared_ptr<arrow::Array> arr = util::WrapNumericVector(data);

  const T *data_ptr = std::static_pointer_cast<arrow::NumericArray<TestType>>(arr)->raw_values();
  REQUIRE(data_ptr == data.data());
  for (size_t i = 0; i < data.size(); i++) {
    REQUIRE(data[i] == *(data_ptr + i));
  }
}

} // namespace test
} // namespace cylon
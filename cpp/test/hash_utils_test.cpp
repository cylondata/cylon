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

#include "test_header.hpp"
#include "test_utils.hpp"

using namespace cylon;

template<typename T>
int dummy_compare(const T &a, const T &b) {
  if (a == b) return 0;
  else if (a < b) return -1;
  else return 1;
}

TEST_CASE("testing hash utils", "[utils]") {

  std::vector<int64_t> v1{10, 20, 50, 40, 30, 30};
  auto arr1 = test::VectorToArrowArray(v1);

  std::vector<int64_t> v2{10, 30, 10};
  auto arr2 = test::VectorToArrowArray(v2);

  std::vector<std::shared_ptr<arrow::Field>> fields
      {std::make_shared<arrow::Field>("a", arrow::int64()), std::make_shared<arrow::Field>("b", arrow::int64())};
  auto schema = std::make_shared<arrow::Schema>(fields);
  const std::shared_ptr<arrow::Table> &t1 = arrow::Table::Make(schema, {arr1, arr1});
  const std::shared_ptr<arrow::Table> &t2 = arrow::Table::Make(schema, {arr2, arr2});

  SECTION("testing TwoNumericRowIndexComparator") {
    const std::shared_ptr<TwoArrayIndexComparator> &comp = CreateTwoArrayIndexComparator(arr1, arr2);

    for (size_t i = 0; i < v1.size(); i++) {
      for (size_t j = 0; j < v2.size(); j++) {
        auto expected = dummy_compare(v1[i], v2[j]);
        auto got = comp->compare(i, util::SetBit(j));
        REQUIRE((got == expected));
      }
    }
  }

  SECTION("testing TwoTableRowIndexEqualTo") {
    TwoTableRowIndexEqualTo equal_to(t1, t2);

    // t1 and t2
    REQUIRE((equal_to(0, util::SetBit(2)) == true));
    REQUIRE((equal_to(0, util::SetBit(1)) == false));

    // within t1
    REQUIRE((equal_to(4, 5) == true));
    REQUIRE((equal_to(3, 4) == false));

    // within t2
    REQUIRE((equal_to(util::SetBit(0), util::SetBit(2)) == true));
    REQUIRE((equal_to(util::SetBit(0), util::SetBit(1)) == false));
  }

  SECTION("testing TwoTableRowIndexHash") {
    TwoTableRowIndexHash hash(t1, t2);

    // t1 and t2
    REQUIRE((hash(0) == hash(util::SetBit(0))));
    REQUIRE((hash(0) != hash(util::SetBit(1))));

    // within t1
    REQUIRE((hash(4) == hash(5)));
    REQUIRE((hash(3) != hash(4)));

    // within t2
    REQUIRE((hash(util::SetBit(0)) == hash(util::SetBit(2))));
    REQUIRE((hash(util::SetBit(0)) != hash(util::SetBit(1))));
  }

}
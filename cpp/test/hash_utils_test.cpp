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

#include "cylon/arrow/arrow_comparator.hpp"
#include "cylon/util/arrow_utils.hpp"
#include "common/test_header.hpp"
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
    std::unique_ptr<DualArrayIndexComparator> comp;
    CHECK_CYLON_STATUS(CreateDualArrayIndexComparator(arr1, arr2, &comp));

    for (size_t i = 0; i < v1.size(); i++) {
      for (size_t j = 0; j < v2.size(); j++) {
        auto expected = dummy_compare(v1[i], v2[j]);
        auto got = comp->compare(i, util::SetBit(j));
        REQUIRE((got == expected));
      }
    }
  }

  SECTION("testing TwoTableRowIndexEqualTo") {
    std::unique_ptr<DualTableRowIndexEqualTo> equal_to;
    CHECK_CYLON_STATUS(DualTableRowIndexEqualTo::Make(t1, t2, &equal_to));

    // t1 and t2
    REQUIRE((equal_to->operator()(0, util::SetBit(2)) == true));
    REQUIRE((equal_to->operator()(0, util::SetBit(1)) == false));

    // within t1
    REQUIRE((equal_to->operator()(4, 5) == true));
    REQUIRE((equal_to->operator()(3, 4) == false));

    // within t2
    REQUIRE((equal_to->operator()(util::SetBit(0), util::SetBit(2)) == true));
    REQUIRE((equal_to->operator()(util::SetBit(0), util::SetBit(1)) == false));
  }

  SECTION("testing TwoTableRowIndexHash") {
    std::unique_ptr<DualTableRowIndexHash> hash;
    CHECK_CYLON_STATUS(DualTableRowIndexHash::Make(t1, t2, &hash));

    // t1 and t2
    REQUIRE((hash->operator()(0) == hash->operator()(util::SetBit(0))));
    REQUIRE((hash->operator()(0) != hash->operator()(util::SetBit(1))));

    // within t1
    REQUIRE((hash->operator()(4) == hash->operator()(5)));
    REQUIRE((hash->operator()(3) != hash->operator()(4)));

    // within t2
    REQUIRE((hash->operator()(util::SetBit(0)) == hash->operator()(util::SetBit(2))));
    REQUIRE((hash->operator()(util::SetBit(0)) != hash->operator()(util::SetBit(1))));
  }

}
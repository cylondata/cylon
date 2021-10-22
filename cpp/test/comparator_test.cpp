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

#include "cylon/arrow/arrow_type_traits.hpp"

namespace cylon {
namespace test {

TEMPLATE_LIST_TEST_CASE("test comparator - numeric", "[comp]", ArrowNumericTypes) {
  using T = typename ArrowTypeTraits<TestType>::ValueT;

  auto type = default_type_instance<TestType>();
  INFO("testing " + type->ToString())

  auto dummy_comp = [](bool asc, T v1, T v2) -> int {
    if (v1 == v2) return 0;
    else if (v1 < v2) return asc ? -1 : 1;
    else return asc ? 1 : -1;
  };

  auto arr = ArrayFromJSON(type, "[1, 2, 3, 4, 5, 6, 7, 1, 2, 3]");
  const T *data = arr->data()->template GetValues<T>(1);

  std::unique_ptr<ArrayIndexComparator> comp;
  for (bool asc: {true, false}) {
    SECTION(asc ? "asc" : "desc") {
      CHECK_CYLON_STATUS(CreateArrayIndexComparator(arr, &comp, asc));
      for (int64_t i = 0; i < arr->length(); i++) {
        for (int64_t j = 0; j < arr->length(); j++) {
          REQUIRE((dummy_comp(asc, data[i], data[j]) == comp->compare(i, j)));
        }
      }
    }
  }

  SECTION("equal to") {
    CHECK_CYLON_STATUS(CreateArrayIndexComparator(arr, &comp));
    for (int64_t i = 0; i < arr->length(); i++) {
      for (int64_t j = 0; j < arr->length(); j++) {
        REQUIRE((data[i] == data[j]) == comp->equal_to(i, j));
      }
    }
  }
}

//TEMPLATE_LIST_TEST_CASE("test comparator w/ null- numeric", "[comp]", ArrowNumericTypes) {
//  using T = typename ArrowTypeTraits<TestType>::ValueT;
//
//  auto type = default_type_instance<TestType>();
//  INFO("testing " + type->ToString())
//
//  auto arr = ArrayFromJSON(type, "[1, 2, 3, 4, null, 6, null, 1, 2, 3]");
//  const T *data = arr->data()->template GetValues<T>(1);
//
//  auto dummy_comp = [](bool asc, int64_t v1, int64_t v2) -> int {
//    if (v1 == v2) return 0;
//    else if (v1 < v2) return asc ? -1 : 1;
//    else return asc ? 1 : -1;
//  };
//
//  std::unique_ptr<ArrayIndexComparator> comp;
//  for (bool asc: {true, false}) {
//    SECTION(asc ? "asc" : "desc") {
//      CHECK_CYLON_STATUS(CreateArrayIndexComparator(arr, &comp, asc));
//      for (int64_t i = 0; i < arr->length(); i++) {
//        for (int64_t j = 0; j < arr->length(); j++) {
//          REQUIRE((dummy_comp(asc, data[i], data[j]) == comp->compare(i, j)));
//        }
//      }
//    }
//  }
//
//  SECTION("equal to") {
//    CHECK_CYLON_STATUS(CreateArrayIndexComparator(arr, &comp));
//    for (int64_t i = 0; i < arr->length(); i++) {
//      for (int64_t j = 0; j < arr->length(); j++) {
//        REQUIRE((data[i] == data[j]) == comp->equal_to(i, j));
//      }
//    }
//  }
//}

}
}
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

template<typename T, typename Enable = void>
struct Helper {};

template<typename T>
struct Helper<T, std::enable_if_t<std::is_arithmetic<T>::value>> {
  static T max() { return std::numeric_limits<T>::max(); }
  static T min() { return std::numeric_limits<T>::min(); }

  static int compare(bool asc, const T &v1, const T &v2) {
    if (v1 == v2) return 0;
    else if (v1 < v2) return asc ? -1 : 1;
    else return asc ? 1 : -1;
  }
};

template<>
struct Helper<arrow::util::string_view> {
  static arrow::util::string_view max() { return "ZZZZ"; }
  static arrow::util::string_view min() { return ""; }

  static int compare(bool asc,
                     const arrow::util::string_view &v1,
                     const arrow::util::string_view &v2) {
    return asc ? v1.compare(v2) : v2.compare(v1);
  }
};

template<typename ArrowT>
void TestArrayIndexComparator(std::string arr_str) {
  using T = typename ArrowTypeTraits<ArrowT>::ValueT;
  using ArrayT = typename ArrowTypeTraits<ArrowT>::ArrayT;

  auto type = default_type_instance<ArrowT>();
  INFO("testing " + type->ToString())

  auto arr = std::static_pointer_cast<ArrayT>(ArrayFromJSON(type, arr_str));

  // dummy comparator to emulate the behavior
  auto dummy_comp = [&](bool null_order, bool asc, int64_t i, int64_t j) -> int {
    T v1 = arr->IsNull(i) ? (null_order ? Helper<T>::max() : Helper<T>::min())
                          : arr->GetView(i);
    T v2 = arr->IsNull(j) ? (null_order ? Helper<T>::max() : Helper<T>::min())
                          : arr->GetView(j);
    return Helper<T>::compare(asc, v1, v2);
  };

  // dummy comparator and comp->compare might give different output ints. for those to be valid,
  // they need to be equal or have the same sign.
  auto check_comp_values = [](int a, int b) {
    if (a == b) return true;
    return a * b > 0;
  };

  std::unique_ptr<ArrayIndexComparator> comp;
  for (bool null_order: {true, false}) {
    SECTION(null_order ? "null to max" : "null to min") {
      for (bool asc: {true, false}) {
        SECTION(asc ? "asc " : "desc ") {
          CHECK_CYLON_STATUS(CreateArrayIndexComparator(arr, &comp, asc, null_order));
          for (int64_t i = 0; i < arr->length(); i++) {
            for (int64_t j = 0; j < arr->length(); j++) {
              auto exp = dummy_comp(null_order, asc, i, j);
              auto got = comp->compare(i, j);
              INFO("" << i << " " << j << " " << exp << " " << got);
              REQUIRE(check_comp_values(exp, got));
            }
          }
        }
      }
    }
  }

  SECTION("equal to") {
    // dummy equal to operator to emulate the behavior
    auto dummy_equal = [&](int64_t i, int64_t j) {
      // if i'th value is null, replace with some known value
      T v1 = arr->IsNull(i) ? Helper<T>::max() : arr->GetView(i);
      T v2 = arr->IsNull(j) ? Helper<T>::max() : arr->GetView(j);
      return v1 == v2;
    };

    CHECK_CYLON_STATUS(CreateArrayIndexComparator(arr, &comp));
    for (int64_t i = 0; i < arr->length(); i++) {
      for (int64_t j = 0; j < arr->length(); j++) {
        REQUIRE((dummy_equal(i, j) == comp->equal_to(i, j)));
      }
    }
  }
}

TEMPLATE_LIST_TEST_CASE("test comparator - numeric", "[comp]", ArrowNumericTypes) {
  TestArrayIndexComparator<TestType>("[10, 2, 3, 4, 5, 6, 7, 10, 2, 3]");
}

TEMPLATE_LIST_TEST_CASE("test comparator w/ null- numeric", "[comp]", ArrowNumericTypes) {
  TestArrayIndexComparator<TestType>("[10, 2, 3, 4, null, 6, null, 10, 2, 3]");
}

TEMPLATE_LIST_TEST_CASE("test comparator - binary", "[comp]", ArrowBinaryTypes) {
  TestArrayIndexComparator<TestType>(
      R"(["10", "2", "3", "4", "5", "6", "7", "10", "2", "3"])");
}

TEMPLATE_LIST_TEST_CASE("test comparator w/ nulls - binary", "[comp]", ArrowBinaryTypes) {
  TestArrayIndexComparator<TestType>(R"(["10", "2", "3", "4", null, "6", null, "10", "2", "3"])");
}

/*
 * Split an array into 2 and check against the results of ArrayIndexComparator
 */
template<typename ArrowT>
void TestDualArrayIndexComparator(std::string arr_str) {
  using ArrayT = typename ArrowTypeTraits<ArrowT>::ArrayT;

  auto type = default_type_instance<ArrowT>();
  INFO("testing " + type->ToString())

  auto arr = std::static_pointer_cast<ArrayT>(ArrayFromJSON(type, arr_str));
  auto arr1 = std::static_pointer_cast<ArrayT>(arr->Slice(0, arr->length() / 2));
  auto arr2 = std::static_pointer_cast<ArrayT>(arr->Slice(arr->length() / 2, arr->length()));

  // dummy comparator and comp->compare might give different output ints. for those to be valid,
  // they need to be equal or have the same sign.
  auto check_comp_values = [](int a, int b) {
    if (a == b) return true;
    return a * b > 0;
  };

  std::unique_ptr<ArrayIndexComparator> exp_comp;
  std::unique_ptr<DualArrayIndexComparator> comp;
  for (bool null_order: {true, false}) {
    SECTION(null_order ? "null to max" : "null to min") {
      for (bool asc: {true, false}) {
        SECTION(asc ? "asc " : "desc ") {
          CHECK_CYLON_STATUS(CreateArrayIndexComparator(arr, &exp_comp, asc, null_order));
          CHECK_CYLON_STATUS(CreateDualArrayIndexComparator(arr1, arr2, &comp, asc, null_order));
          for (int64_t i = 0; i < arr1->length(); i++) {
            for (int64_t j = 0; j < arr2->length(); j++) {
              auto exp = exp_comp->compare(i, arr1->length() + j);
              auto got = comp->compare(i, util::SetBit(j));
              INFO("" << i << " " << j << " " << exp << " " << got);
              REQUIRE(check_comp_values(exp, got));
            }
          }
        }
      }
    }
  }

  SECTION("equal to") {
    CHECK_CYLON_STATUS(CreateArrayIndexComparator(arr, &exp_comp));
    CHECK_CYLON_STATUS(CreateDualArrayIndexComparator(arr1, arr2, &comp));
    for (int64_t i = 0; i < arr1->length(); i++) {
      for (int64_t j = 0; j < arr2->length(); j++) {
        auto exp = exp_comp->equal_to(i, arr1->length() + j);
        auto got = comp->equal_to(i, util::SetBit(j));
        REQUIRE(exp == got);
      }
    }
  }
}

TEMPLATE_LIST_TEST_CASE("test dual comparator - numeric", "[comp]", ArrowNumericTypes) {
  TestDualArrayIndexComparator<TestType>(
      "[10, 2, 3, 4, 5, 6, 7, 10, 2, 3, 3, 4, 5, 6, 7, 10, 2, 3, 3]");
}

TEMPLATE_LIST_TEST_CASE("test dual comparator w/ null- numeric", "[comp]", ArrowNumericTypes) {
  TestDualArrayIndexComparator<TestType>(
      "[10, 2, 3, 4, null, 6, null, 10, 2, 3, 3, 4, null, 6, null, 10]");
}

TEMPLATE_LIST_TEST_CASE("test dual comparator - binary", "[comp]", ArrowBinaryTypes) {
  TestDualArrayIndexComparator<TestType>(
      R"(["10", "2", "3", "4", "5", "6", "7", "10", "2", "3", "4", "5", "6", "7", "10", "2", "3"])");
}

TEMPLATE_LIST_TEST_CASE("test dual comparator w/ nulls - binary", "[comp]", ArrowBinaryTypes) {
  TestDualArrayIndexComparator<TestType>(
      R"(["10", "2", "3", "4", null, "6", null, "10", "2", "3", "4", null, "6", null, "10", "2", "3"])");
}

}
}
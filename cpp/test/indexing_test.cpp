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

#include "cylon/indexing/index.hpp"
#include "common/test_header.hpp"
#include "test_index_utils.hpp"
#include "test_arrow_utils.hpp"
#include "test_utils.hpp"
#include "test_macros.hpp"

#include <arrow/ipc/api.h>

//using namespace cylon;

//TEST_CASE("Index testing", "[indexing]") {
//std::string path1 = "../data/input/indexing_data.csv";
//
//std::vector<std::string> output_files {"../data/output/indexing_loc_hl_",
//                                       "../data/output/indexing_loc_hl_",
//                                       "../data/output/indexing_loc_r_"};
//
//
//std::vector<cylon::IndexingType> indexing_types = {cylon::IndexingType::Hash,
//                                                       cylon::IndexingType::Linear,
//                                                       cylon::IndexingType::Range};
//
//SECTION("testing build index") {
//  for(auto type : indexing_types) {
//    REQUIRE(test::TestIndexBuildOperation(path1, type) == 0);
//  }
//}
//SECTION("testing loc index 1") {
//for(size_t i=0; i < output_files.size(); i++) {
//  auto type = indexing_types.at(i);
//  auto output_file = output_files.at(i);
//  REQUIRE(test::TestIndexLocOperation1(path1, type, output_file) == 0);
//}
//
//SECTION("testing loc index 2") {
//for(size_t i = 0; i<output_files.size(); i++) {
//  auto type = indexing_types.at(i);
//  auto output_file = output_files.at(i);
//  REQUIRE(test::TestIndexLocOperation2(path1, type, output_file) == 0);
//}
//}
//
//SECTION("testing loc index 3") {
//  for(size_t i = 0; i<output_files.size(); i++) {
//    auto type = indexing_types.at(i);
//    auto output_file = output_files.at(i);
//    REQUIRE(test::TestIndexLocOperation3(path1, type, output_file) == 0);
//  }
//}
//
//SECTION("testing loc index 4") {
//  for(size_t i = 0; i<output_files.size(); i++) {
//    auto type = indexing_types.at(i);
//    auto output_file = output_files.at(i);
//    REQUIRE(test::TestIndexLocOperation4(path1, type, output_file) == 0);
//  }
//}
//
//SECTION("testing loc index 5") {
//  for(size_t i = 0; i<output_files.size(); i++) {
//    auto type = indexing_types.at(i);
//    auto output_file = output_files.at(i);
//    REQUIRE(test::TestIndexLocOperation5(path1, type, output_file) == 0);
//  }
//}
//
//SECTION("testing loc index 6") {
//  for(size_t i = 0; i<output_files.size(); i++) {
//    auto type = indexing_types.at(i);
//    auto output_file = output_files.at(i);
//    REQUIRE(test::TestIndexLocOperation6(path1, type, output_file) == 0);
//  }
//}
//
//}
//
//}

namespace cylon {
namespace test {

void TestIndex(IndexingType index_type, const std::shared_ptr<arrow::DataType> &data_type) {
  const auto &idx_arr = ArrayFromJSON(data_type, "[1, 2, 2, null, 5, 5, null, 8]");

  std::shared_ptr<BaseArrowIndex> index;
  CHECK_CYLON_STATUS(BuildIndex(idx_arr, index_type, &index));

  SECTION("check LocationByValue - all locations") {
    std::shared_ptr<arrow::Int64Array> locs;

    SECTION("check non-null value") {
      auto val = idx_arr->GetScalar(1).ValueOrDie(); // get idx_arr[1] (= 2)
      CHECK_CYLON_STATUS(index->LocationByValue(val, &locs));
      CHECK_ARRAYS_EQUAL(ArrayFromJSON(arrow::int64(), "[1, 2]"), locs);
    }

    SECTION("check null value") {
      auto val = idx_arr->GetScalar(3).ValueOrDie(); // get idx_arr[3] (= null)
      CHECK_CYLON_STATUS(index->LocationByValue(val, &locs));
      CHECK_ARRAYS_EQUAL(ArrayFromJSON(arrow::int64(), "[3, 6]"), locs);
    }

    SECTION("check non-existent value") {
      auto val = arrow::MakeScalar(data_type, 10).ValueOrDie();
      REQUIRE_FALSE(index->LocationByValue(val, &locs).is_ok());
    }
  }

  SECTION("check LocationByValue - first location") {
    int64_t loc = -1;

    SECTION("check non-null value") {
      auto val = idx_arr->GetScalar(1).ValueOrDie(); // get idx_arr[1] (= 2)
      CHECK_CYLON_STATUS(index->LocationByValue(val, &loc));
      REQUIRE(loc == 1);
    }

    SECTION("check null value") {
      auto val = idx_arr->GetScalar(3).ValueOrDie(); // get idx_arr[3] (= null)
      CHECK_CYLON_STATUS(index->LocationByValue(val, &loc));
      REQUIRE(loc == 3);
    }

    SECTION("check non-existent value") {
      auto val = arrow::MakeScalar(data_type, 10).ValueOrDie();
      REQUIRE_FALSE(index->LocationByValue(val, &loc).is_ok());
    }
  }

  SECTION("check LocationByVector") {
    std::shared_ptr<arrow::Int64Array> locs;

    // todo: this method behaves differently for Linear and Hash indices!
    if (index_type == Linear) {
      SECTION("check all non-null values") {
        auto val = ArrayFromJSON(data_type, "[1, 2]");
        CHECK_CYLON_STATUS(index->LocationByVector(val, &locs));
        CHECK_ARRAYS_EQUAL(ArrayFromJSON(arrow::int64(), "[0, 1, 2]"), locs);
      }

      SECTION("check with a null value") {
        auto val = ArrayFromJSON(data_type, "[2, null, 5]");
        CHECK_CYLON_STATUS(index->LocationByVector(val, &locs));
        CHECK_ARRAYS_EQUAL(ArrayFromJSON(arrow::int64(), "[1, 2, 3, 4, 5, 6]"), locs);
      }

      SECTION("Linear index fails with duplicate search values") {
        auto val = ArrayFromJSON(data_type, "[1, 2, 1]");
        REQUIRE_FALSE(index->LocationByVector(val, &locs).is_ok());
      }
    } else if (index_type == Hash) {
      SECTION("check all non-null values") {
        auto val = ArrayFromJSON(data_type, "[1, 2, 1]");
        CHECK_CYLON_STATUS(index->LocationByVector(val, &locs));
        CHECK_ARRAYS_EQUAL(ArrayFromJSON(arrow::int64(), "[0, 1, 2, 0]"), locs);
      }

      SECTION("check with a null value") {
        auto val = ArrayFromJSON(data_type, "[2, null, 5]");
        CHECK_CYLON_STATUS(index->LocationByVector(val, &locs));
        CHECK_ARRAYS_EQUAL(ArrayFromJSON(arrow::int64(), "[1, 2, 3, 6, 4, 5]"), locs);
      }
    }

    SECTION("check non-existent values") {
      auto val = ArrayFromJSON(data_type, "[10, 11]");
      REQUIRE_FALSE(index->LocationByVector(val, &locs).is_ok());
    }
  }

  SECTION("check LocationRangeByValue") {
    int64_t start = -1, end = -1;

    SECTION("check non-null value") {
      auto startv = arrow::MakeScalar(data_type, 1).ValueOrDie();
      auto endv = arrow::MakeScalar(data_type, 2).ValueOrDie();
      CHECK_CYLON_STATUS(index->LocationRangeByValue(startv, endv, &start, &end));
      REQUIRE((start == 0 && end == 2));
    }

    SECTION("check with null value") {
      auto startv = arrow::MakeScalar(data_type, 2).ValueOrDie();
      auto endv = arrow::MakeScalar(data_type, 5).ValueOrDie();
      CHECK_CYLON_STATUS(index->LocationRangeByValue(startv, endv, &start, &end));
      REQUIRE((start == 1 && end == 5));
    }

    SECTION("fail for non-contiguous ranges") {
      auto startv = arrow::MakeScalar(data_type, 2).ValueOrDie();
      auto endv = arrow::MakeNullScalar(data_type);
      REQUIRE_FALSE(index->LocationRangeByValue(startv, endv, &start, &end).is_ok());
    }
  }
}

using ArrowTypes = std::tuple<arrow::Int8Type, arrow::Int16Type, arrow::Int32Type, arrow::Int64Type,
                              arrow::UInt8Type, arrow::UInt16Type, arrow::UInt32Type, arrow::UInt64Type,
                              arrow::FloatType, arrow::DoubleType>;

TEMPLATE_LIST_TEST_CASE("Test linear index", "[indexing]", ArrowTypes) {
  auto type = arrow::TypeTraits<TestType>::type_singleton();
  TestIndex(IndexingType::Linear, type);
}

TEMPLATE_LIST_TEST_CASE("Test hash index", "[indexing]", ArrowTypes) {
  auto type = arrow::TypeTraits<TestType>::type_singleton();
  TestIndex(IndexingType::Hash, type);
}

} // namespace test
} // namespace cylon

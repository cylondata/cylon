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
#include "test_macros.hpp"

#include <arrow/ipc/api.h>

namespace cylon {
namespace test {

TEST_CASE("Index testing", "[indexing]") {
  std::string path1 = "../data/input/indexing_data.csv";
  std::unordered_map<IndexingType, std::string> out_files{
      {IndexingType::Hash, "../data/output/indexing_loc_hl_"},
      {IndexingType::Linear, "../data/output/indexing_loc_hl_"},
      {IndexingType::Range, "../data/output/indexing_loc_r_"}
  };

  SECTION("testing build index") {
    for (auto & item : out_files) {
      TestIndexBuildOperation(ctx, path1, item.first);
    }
  }

  SECTION("testing loc index 1") {
    for (auto & item : out_files) {
      TestIndexLocOperation1(ctx, path1, item.first, item.second);
    }
  }

  SECTION("testing loc index 2") {
    for (auto & item : out_files) {
      TestIndexLocOperation2(ctx, path1, item.first, item.second);
    }
  }

  SECTION("testing loc index 3") {
    for (auto & item : out_files) {
      TestIndexLocOperation3(ctx, path1, item.first, item.second);
    }
  }

  SECTION("testing loc index 4") {
    for (auto & item : out_files) {
      TestIndexLocOperation4(ctx, path1, item.first, item.second);
    }
  }

  SECTION("testing loc index 5") {
    for (auto & item : out_files) {
      TestIndexLocOperation5(ctx, path1, item.first, item.second);
    }
  }

  SECTION("testing loc index 6") {
    for (auto & item : out_files) {
      TestIndexLocOperation6(ctx, path1, item.first, item.second);
    }
  }
}

TEST_CASE("Test range index", "[indexing]") {
  const auto &index = BuildRangeIndex(nullptr, 0, 10, 1);

  auto data_type = arrow::int64();

  SECTION("check LocationByValue - all locations") {
    std::shared_ptr<arrow::Int64Array> locs;

    SECTION("check non-null value") {
      auto val = arrow::MakeScalar(data_type, 2).ValueOrDie();
      CHECK_CYLON_STATUS(index->LocationByValue(val, &locs));
      CHECK_ARRAYS_EQUAL(ArrayFromJSON(arrow::int64(), "[2]"), locs);
    }

    SECTION("check null value") {
      auto val = arrow::MakeNullScalar(data_type);
      EXPECT_FAIL_WITH_MSG(Code::KeyError, "null", index->LocationByValue(val, &locs));
    }

    SECTION("check non-existent value") {
      auto val = arrow::MakeScalar(data_type, 10).ValueOrDie();
      EXPECT_FAIL_WITH_MSG(Code::KeyError, "not found", index->LocationByValue(val, &locs));
    }
  }

  SECTION("check LocationByValue - first location") {
    int64_t loc = -1;

    SECTION("check non-null value") {
      auto val = arrow::MakeScalar(data_type, 2).ValueOrDie();
      CHECK_CYLON_STATUS(index->LocationByValue(val, &loc));
      REQUIRE(loc == 2);
    }

    SECTION("check null value") {
      auto val = arrow::MakeNullScalar(data_type);
      EXPECT_FAIL_WITH_MSG(Code::KeyError, "null", index->LocationByValue(val, &loc));
    }

    SECTION("check non-existent value") {
      auto val = arrow::MakeScalar(data_type, 10).ValueOrDie();
      EXPECT_FAIL_WITH_MSG(Code::KeyError, "not found", index->LocationByValue(val, &loc));
    }
  }

  SECTION("check LocationByVector") {
    std::shared_ptr<arrow::Int64Array> locs;

    SECTION("check all non-null values") {
      auto val = ArrayFromJSON(data_type, "[1, 2]");
      CHECK_CYLON_STATUS(index->LocationByVector(val, &locs));
      CHECK_ARRAYS_EQUAL(ArrayFromJSON(arrow::int64(), "[1, 2]"), locs);
    }

    SECTION("check with a null value") {
      auto val = ArrayFromJSON(data_type, "[2, null, 5]");
      EXPECT_FAIL_WITH_MSG(Code::KeyError, "null", index->LocationByVector(val, &locs));
    }

    SECTION("Linear index fails with duplicate search values") {
      auto val = ArrayFromJSON(data_type, "[1, 2, 1]");
      CHECK_CYLON_STATUS(index->LocationByVector(val, &locs));
      CHECK_ARRAYS_EQUAL(ArrayFromJSON(arrow::int64(), "[1, 2, 1]"), locs);
    }
  }

  SECTION("check LocationRangeByValue") {
    int64_t start = -1, end = -1;

    SECTION("check non-null value") {
      auto startv = arrow::MakeScalar(data_type, 1).ValueOrDie();
      auto endv = arrow::MakeScalar(data_type, 2).ValueOrDie();
      CHECK_CYLON_STATUS(index->LocationRangeByValue(startv, endv, &start, &end));
      REQUIRE((start == 1 && end == 2));
    }

    SECTION("check with null value") {
      auto startv = arrow::MakeScalar(data_type, 2).ValueOrDie();
      auto endv = arrow::MakeScalar(data_type, 5).ValueOrDie();
      CHECK_CYLON_STATUS(index->LocationRangeByValue(startv, endv, &start, &end));
      REQUIRE((start == 2 && end == 5));
    }
  }
}

template<typename ArrowT>
arrow::enable_if_has_c_type<ArrowT, void> TestIndexNumeric(IndexingType index_type) {
  auto data_type = default_type_instance<ArrowT>();
  const auto &idx_arr = ArrayFromJSON(data_type, "[1, 2, 2, null, 5, 5, null, 8]");

  std::shared_ptr<BaseArrowIndex> index;
  CHECK_CYLON_STATUS(BuildIndex(idx_arr, index_type, &index));

  SECTION("check LocationByValue - all locations") {
    std::shared_ptr<arrow::Int64Array> locs;

    SECTION("check non-null value") {
      auto val = arrow::MakeScalar(data_type, 2).ValueOrDie();
      CHECK_CYLON_STATUS(index->LocationByValue(val, &locs));
      CHECK_ARRAYS_EQUAL(ArrayFromJSON(arrow::int64(), "[1, 2]"), locs);
    }

    SECTION("check null value") {
      auto val = arrow::MakeNullScalar(data_type);
      CHECK_CYLON_STATUS(index->LocationByValue(val, &locs));
      CHECK_ARRAYS_EQUAL(ArrayFromJSON(arrow::int64(), "[3, 6]"), locs);
    }

    SECTION("check non-existent value") {
      auto val = arrow::MakeScalar(data_type, 10).ValueOrDie();
      EXPECT_FAIL_WITH_MSG(Code::KeyError, "not found", index->LocationByValue(val, &locs));
    }
  }

  SECTION("check LocationByValue - first location") {
    int64_t loc = -1;

    SECTION("check non-null value") {
      auto val = arrow::MakeScalar(data_type, 2).ValueOrDie();
      CHECK_CYLON_STATUS(index->LocationByValue(val, &loc));
      REQUIRE(loc == 1);
    }

    SECTION("check null value") {
      auto val = arrow::MakeNullScalar(data_type);
      CHECK_CYLON_STATUS(index->LocationByValue(val, &loc));
      REQUIRE(loc == 3);
    }

    SECTION("check non-existent value") {
      auto val = arrow::MakeScalar(data_type, 10).ValueOrDie();
      EXPECT_FAIL_WITH_MSG(Code::KeyError, "not found", index->LocationByValue(val, &loc));
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
      EXPECT_FAIL_WITH_MSG(Code::KeyError, "not found", index->LocationByVector(val, &locs));
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
      EXPECT_FAIL_WITH_MSG(Code::KeyError, "non-unique key",
                           index->LocationRangeByValue(startv, endv, &start, &end));
    }
  }
}

template<typename ArrowT>
arrow::enable_if_base_binary<ArrowT, void> TestIndexBinary(IndexingType index_type) {
  auto data_type = default_type_instance<ArrowT>();
  const auto &idx_arr = ArrayFromJSON(data_type, R"(["a", "b", "b", null, "c", "c", null, "d"])");

  std::shared_ptr<BaseArrowIndex> index;
  CHECK_CYLON_STATUS(BuildIndex(idx_arr, index_type, &index));

  SECTION("check LocationByValue - all locations") {
    std::shared_ptr<arrow::Int64Array> locs;

    SECTION("check non-null value") {
      auto val = MakeBinaryScalar<ArrowT>("b");
      CHECK_CYLON_STATUS(index->LocationByValue(val, &locs));
      CHECK_ARRAYS_EQUAL(ArrayFromJSON(arrow::int64(), "[1, 2]"), locs);
    }

    SECTION("check null value") {
      auto val = arrow::MakeNullScalar(data_type);
      CHECK_CYLON_STATUS(index->LocationByValue(val, &locs));
      CHECK_ARRAYS_EQUAL(ArrayFromJSON(arrow::int64(), "[3, 6]"), locs);
    }

    SECTION("check non-existent value") {
      auto val = MakeBinaryScalar<ArrowT>("e");
      EXPECT_FAIL_WITH_MSG(Code::KeyError, "not found", index->LocationByValue(val, &locs));
    }
  }

  SECTION("check LocationByValue - first location") {
    int64_t loc = -1;

    SECTION("check non-null value") {
      auto val = MakeBinaryScalar<ArrowT>("b");
      CHECK_CYLON_STATUS(index->LocationByValue(val, &loc));
      REQUIRE(loc == 1);
    }

    SECTION("check null value") {
      auto val = arrow::MakeNullScalar(data_type);
      CHECK_CYLON_STATUS(index->LocationByValue(val, &loc));
      REQUIRE(loc == 3);
    }

    SECTION("check non-existent value") {
      auto val = MakeBinaryScalar<ArrowT>("e");
      EXPECT_FAIL_WITH_MSG(Code::KeyError, "not found", index->LocationByValue(val, &loc));
    }
  }

  SECTION("check LocationByVector") {
    std::shared_ptr<arrow::Int64Array> locs;

    // todo: this method behaves differently for Linear and Hash indices!
    if (index_type == Linear) {
      SECTION("check all non-null values") {
        auto val = ArrayFromJSON(data_type, R"(["a", "b"])");
        CHECK_CYLON_STATUS(index->LocationByVector(val, &locs));
        CHECK_ARRAYS_EQUAL(ArrayFromJSON(arrow::int64(), "[0, 1, 2]"), locs);
      }

      SECTION("check with a null value") {
        auto val = ArrayFromJSON(data_type, R"(["b", null, "c"])");
        CHECK_CYLON_STATUS(index->LocationByVector(val, &locs));
        CHECK_ARRAYS_EQUAL(ArrayFromJSON(arrow::int64(), "[1, 2, 3, 4, 5, 6]"), locs);
      }

      SECTION("Linear index fails with duplicate search values") {
        auto val = ArrayFromJSON(data_type, R"(["a", "b", "a"])");
        EXPECT_FAIL_WITH_MSG(Code::Invalid, "duplicate", index->LocationByVector(val, &locs));
      }
    } else if (index_type == Hash) {
      SECTION("check all non-null values") {
        auto val = ArrayFromJSON(data_type, R"(["a", "b", "a"])");
        CHECK_CYLON_STATUS(index->LocationByVector(val, &locs));
        CHECK_ARRAYS_EQUAL(ArrayFromJSON(arrow::int64(), "[0, 1, 2, 0]"), locs);
      }

      SECTION("check with a null value") {
        auto val = ArrayFromJSON(data_type, R"(["b", null, "c"])");
        CHECK_CYLON_STATUS(index->LocationByVector(val, &locs));
        CHECK_ARRAYS_EQUAL(ArrayFromJSON(arrow::int64(), "[1, 2, 3, 6, 4, 5]"), locs);
      }
    }

    SECTION("check non-existent values") {
      auto val = ArrayFromJSON(data_type, R"(["e", "f"])");
      EXPECT_FAIL_WITH_MSG(Code::KeyError, "not found", index->LocationByVector(val, &locs));
    }
  }

  SECTION("check LocationRangeByValue") {
    int64_t start = -1, end = -1;

    SECTION("check non-null value") {
      auto startv = MakeBinaryScalar<ArrowT>("a");
      auto endv = MakeBinaryScalar<ArrowT>("b");
      CHECK_CYLON_STATUS(index->LocationRangeByValue(startv, endv, &start, &end));
      REQUIRE((start == 0 && end == 2));
    }

    SECTION("check with null value") {
      auto startv = MakeBinaryScalar<ArrowT>("b");
      auto endv = MakeBinaryScalar<ArrowT>("c");
      CHECK_CYLON_STATUS(index->LocationRangeByValue(startv, endv, &start, &end));
      REQUIRE((start == 1 && end == 5));
    }

    SECTION("fail for non-contiguous ranges") {
      auto startv = MakeBinaryScalar<ArrowT>("b");
      auto endv = arrow::MakeNullScalar(data_type);
      EXPECT_FAIL_WITH_MSG(Code::KeyError, "non-unique key",
                           index->LocationRangeByValue(startv, endv, &start, &end));
    }
  }
}

TEMPLATE_LIST_TEST_CASE("Test linear index - numeric types", "[indexing]", ArrowNumericTypes) {
  TestIndexNumeric<TestType>(IndexingType::Linear);
}

TEMPLATE_LIST_TEST_CASE("Test hash index - numeric types", "[indexing]", ArrowNumericTypes) {
  TestIndexNumeric<TestType>(IndexingType::Hash);
}

TEMPLATE_LIST_TEST_CASE("Test linear index - string types", "[indexing]", ArrowBinaryTypes) {
  TestIndexBinary<TestType>(IndexingType::Linear);
}

TEMPLATE_LIST_TEST_CASE("Test hash index - string types", "[indexing]", ArrowBinaryTypes) {
  TestIndexBinary<TestType>(IndexingType::Hash);
}

TEMPLATE_LIST_TEST_CASE("Test linear index - temporal types", "[indexing]", ArrowTemporalTypes) {
  TestIndexNumeric<TestType>(IndexingType::Linear);
}

TEMPLATE_LIST_TEST_CASE("Test hash index - temporal types", "[indexing]", ArrowTemporalTypes) {
  TestIndexNumeric<TestType>(IndexingType::Hash);
}

} // namespace test
} // namespace cylon

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
#include "cylon/indexing/index_utils.hpp"
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
  const auto &index = BuildRangeIndex(0, 10, 1);

  auto data_type = arrow::int64();

  SECTION("check LocationByValue - all locations") {
    std::shared_ptr<arrow::Int64Array> locs;

    SECTION("check non-null value") {
      auto val = arrow::MakeScalar(data_type, 2).ValueOrDie();
      CHECK_CYLON_STATUS(index->LocationByValue(val, &locs));
      CHECK_ARROW_EQUAL(ArrayFromJSON(arrow::int64(), "[2]"), locs);
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
      CHECK_ARROW_EQUAL(ArrayFromJSON(arrow::int64(), "[1, 2]"), locs);
    }

    SECTION("check with a null value") {
      auto val = ArrayFromJSON(data_type, "[2, null, 5]");
      EXPECT_FAIL_WITH_MSG(Code::KeyError, "null", index->LocationByVector(val, &locs));
    }

    SECTION("Linear index fails with duplicate search values") {
      auto val = ArrayFromJSON(data_type, "[1, 2, 1]");
      CHECK_CYLON_STATUS(index->LocationByVector(val, &locs));
      CHECK_ARROW_EQUAL(ArrayFromJSON(arrow::int64(), "[1, 2, 1]"), locs);
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
  auto a_table = arrow::Table::Make(arrow::schema({field("a", data_type)}),
                                    {idx_arr});

  auto table = std::make_shared<Table>(ctx, std::move(a_table));
  CHECK_CYLON_STATUS(table->SetArrowIndex(0, index_type));
  const auto &index = table->GetArrowIndex();

  SECTION("check LocationByValue - all locations") {
    std::shared_ptr<arrow::Int64Array> locs;

    SECTION("check non-null value") {
      auto val = arrow::MakeScalar(data_type, 2).ValueOrDie();
      CHECK_CYLON_STATUS(index->LocationByValue(val, &locs));
      CHECK_ARROW_EQUAL(ArrayFromJSON(arrow::int64(), "[1, 2]"), locs);
    }

    SECTION("check null value") {
      auto val = arrow::MakeNullScalar(data_type);
      CHECK_CYLON_STATUS(index->LocationByValue(val, &locs));
      CHECK_ARROW_EQUAL(ArrayFromJSON(arrow::int64(), "[3, 6]"), locs);
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
        CHECK_ARROW_EQUAL(ArrayFromJSON(arrow::int64(), "[0, 1, 2]"), locs);
      }

      SECTION("check with a null value") {
        auto val = ArrayFromJSON(data_type, "[2, null, 5]");
        CHECK_CYLON_STATUS(index->LocationByVector(val, &locs));
        CHECK_ARROW_EQUAL(ArrayFromJSON(arrow::int64(), "[1, 2, 3, 4, 5, 6]"), locs);
      }

      SECTION("Linear index fails with duplicate search values") {
        auto val = ArrayFromJSON(data_type, "[1, 2, 1]");
        REQUIRE_FALSE(index->LocationByVector(val, &locs).is_ok());
      }
    } else if (index_type == Hash) {
      SECTION("check all non-null values") {
        auto val = ArrayFromJSON(data_type, "[1, 2, 1]");
        CHECK_CYLON_STATUS(index->LocationByVector(val, &locs));
        CHECK_ARROW_EQUAL(ArrayFromJSON(arrow::int64(), "[0, 1, 2, 0]"), locs);
      }

      SECTION("check with a null value") {
        auto val = ArrayFromJSON(data_type, "[2, null, 5]");
        CHECK_CYLON_STATUS(index->LocationByVector(val, &locs));
        CHECK_ARROW_EQUAL(ArrayFromJSON(arrow::int64(), "[1, 2, 3, 6, 4, 5]"), locs);
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

  auto a_table = arrow::Table::Make(arrow::schema({field("a", data_type)}), {idx_arr});

  auto table = std::make_shared<Table>(ctx, std::move(a_table));
  CHECK_CYLON_STATUS(table->SetArrowIndex(0, index_type));
  const auto &index = table->GetArrowIndex();

  SECTION("check LocationByValue - all locations") {
    std::shared_ptr<arrow::Int64Array> locs;

    SECTION("check non-null value") {
      auto val = MakeBinaryScalar<ArrowT>("b");
      CHECK_CYLON_STATUS(index->LocationByValue(val, &locs));
      CHECK_ARROW_EQUAL(ArrayFromJSON(arrow::int64(), "[1, 2]"), locs);
    }

    SECTION("check null value") {
      auto val = arrow::MakeNullScalar(data_type);
      CHECK_CYLON_STATUS(index->LocationByValue(val, &locs));
      CHECK_ARROW_EQUAL(ArrayFromJSON(arrow::int64(), "[3, 6]"), locs);
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
        CHECK_ARROW_EQUAL(ArrayFromJSON(arrow::int64(), "[0, 1, 2]"), locs);
      }

      SECTION("check with a null value") {
        auto val = ArrayFromJSON(data_type, R"(["b", null, "c"])");
        CHECK_CYLON_STATUS(index->LocationByVector(val, &locs));
        CHECK_ARROW_EQUAL(ArrayFromJSON(arrow::int64(), "[1, 2, 3, 4, 5, 6]"), locs);
      }

      SECTION("Linear index fails with duplicate search values") {
        auto val = ArrayFromJSON(data_type, R"(["a", "b", "a"])");
        EXPECT_FAIL_WITH_MSG(Code::Invalid, "duplicate", index->LocationByVector(val, &locs));
      }
    } else if (index_type == Hash) {
      SECTION("check all non-null values") {
        auto val = ArrayFromJSON(data_type, R"(["a", "b", "a"])");
        CHECK_CYLON_STATUS(index->LocationByVector(val, &locs));
        CHECK_ARROW_EQUAL(ArrayFromJSON(arrow::int64(), "[0, 1, 2, 0]"), locs);
      }

      SECTION("check with a null value") {
        auto val = ArrayFromJSON(data_type, R"(["b", null, "c"])");
        CHECK_CYLON_STATUS(index->LocationByVector(val, &locs));
        CHECK_ARROW_EQUAL(ArrayFromJSON(arrow::int64(), "[1, 2, 3, 6, 4, 5]"), locs);
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

TEMPLATE_LIST_TEST_CASE("Test SetIndex ResetIndex", "[indexing]", ArrowNumericTypes) {
  auto type = default_type_instance<TestType>();
  auto ab = TableFromJSON(arrow::schema({
                                            {field("a", type)},
                                            {field("b", arrow::uint32())},
                                        }),
                          {R"([
                                      {"a": null, "b": 5},
                                      {"a": 1,    "b": 3},
                                      {"a": 3,    "b": null},
                                      {"a": 2,    "b": 5},
                                      {"a": 2,    "b": 6},
                                      {"a": 1,    "b": 5}
                                    ])"});
  auto just_a = ab->SelectColumns({0}).ValueOrDie();

  auto col_a = ab->column(0)->chunk(0);
  auto col_b = ab->column(1)->chunk(0);

  SECTION("check default range index") {
    std::shared_ptr<Table> table;
    CHECK_CYLON_STATUS(Table::FromArrowTable(ctx, ab, table));

    const auto &index = table->GetArrowIndex();
    REQUIRE(index->GetIndexingType() == IndexingType::Range);
    REQUIRE(index->size() == table->Rows());
  }

  SECTION("SetIndex + ResetIndex") {
    for (auto index_type: {IndexingType::Linear, IndexingType::Hash}) {
      SECTION("Indexing type " + std::to_string(index_type)) {
        std::shared_ptr<Table> table;
        CHECK_CYLON_STATUS(Table::FromArrowTable(ctx, ab, table));

//        std::shared_ptr<BaseArrowIndex> new_index;
//        CHECK_CYLON_STATUS(BuildIndex(table->get_table(), 0, index_type, &new_index));

        // set index
        CHECK_CYLON_STATUS(table->SetArrowIndex(0, index_type));
        REQUIRE(table->GetArrowIndex()->GetIndexingType() == index_type);
        CHECK_ARROW_EQUAL(ab, table->get_table());

//        // set index with dropping
//        CHECK_CYLON_STATUS(table->SetArrowIndex(new_index, /*drop_index=*/true));
//        REQUIRE(table->GetArrowIndex()->GetIndexingType() == index_type);
//        CHECK_ARROW_EQUAL(just_b, table->get_table());

        // now reset the index
        CHECK_CYLON_STATUS(table->ResetArrowIndex(/*drop=*/false));
        REQUIRE(table->GetArrowIndex()->GetIndexingType() == IndexingType::Range);
        CHECK_ARROW_EQUAL(ab, table->get_table());

        // set index to column b
        CHECK_CYLON_STATUS(table->SetArrowIndex(1, index_type));
        CHECK_ARROW_EQUAL(ab, table->get_table());

        // now reset index with drop. this would not add the column back to the table.
        CHECK_CYLON_STATUS(table->ResetArrowIndex(/*drop=*/true));
        REQUIRE(table->GetArrowIndex()->GetIndexingType() == IndexingType::Range);
        CHECK_ARROW_EQUAL(just_a, table->get_table());
      }
    }
  }
}

TEST_CASE("Test SliceTableByRange", "[indexing]") {
  auto ab = TableFromJSON(arrow::schema({
                                            {field("a", arrow::uint32())},
                                            {field("b", arrow::uint32())},
                                        }),
                          {R"([
                                      {"a": null, "b": 5},
                                      {"a": 1,    "b": 3},
                                      {"a": 3,    "b": null},
                                      {"a": 2,    "b": 5},
                                      {"a": 2,    "b": 6},
                                      {"a": 1,    "b": 5}
                                    ])"});
  std::shared_ptr<Table> table, out;
  CHECK_CYLON_STATUS(Table::FromArrowTable(ctx, ab, table));

  SECTION("Slice - range index") {
    // out has range index by default
    CHECK_CYLON_STATUS(indexing::SliceTableByRange(table, 1, 3, {1}, &out, false));
    auto expected = TableFromJSON(arrow::schema({
                                                    {field("b", arrow::uint32())},
                                                }),
                                  {R"([
                                      {"b": 3},
                                      {"b": null},
                                      {"b": 5}
                                     ])"});
    CHECK_ARROW_EQUAL(expected, out->get_table());

    auto index = std::static_pointer_cast<ArrowRangeIndex>(out->GetArrowIndex());
    REQUIRE(index->start_ == 1);
    REQUIRE(index->end_ == 4);
    REQUIRE(index->step_ == 1);

    int64_t test_loc = -1;
    CHECK_CYLON_STATUS(index->LocationByValue(arrow::MakeScalar<int64_t>(2), &test_loc));
    REQUIRE(test_loc == 1);
  }

  for (auto index_type: {IndexingType::Linear, IndexingType::Hash}) {
    SECTION("Slice - " + std::to_string(index_type)) {
      // set index by column b
      CHECK_CYLON_STATUS(table->SetArrowIndex(1, index_type));

      // select column a --> this would add col b back into the table at the end
      CHECK_CYLON_STATUS(indexing::SliceTableByRange(table, 1, 3, {0}, &out, false));
      auto expected = TableFromJSON(arrow::schema({
                                                      {field("a", arrow::uint32())},
                                                      {field("b", arrow::uint32())},
                                                  }),
                                    {R"([
                                      {"a": 1,    "b": 3},
                                      {"a": 3,    "b": null},
                                      {"a": 2,    "b": 5}
                                     ])"});
      CHECK_ARROW_EQUAL(expected, out->get_table());

      int64_t test_loc = -1;
      CHECK_CYLON_STATUS(out->GetArrowIndex()->LocationByValue(arrow::MakeScalar<uint32_t>(5), &test_loc));
      REQUIRE(test_loc == 2);
      CHECK_CYLON_STATUS(out->GetArrowIndex()->LocationByValue(arrow::MakeNullScalar(arrow::uint32()), &test_loc));
      REQUIRE(test_loc == 1);

      CHECK_CYLON_STATUS(table->ResetArrowIndex()); // put the index array back into the table
    }
  }
}

TEST_CASE("Test FilterTable", "[indexing]") {
  auto ab = TableFromJSON(arrow::schema({
                                            {field("a", arrow::uint32())},
                                            {field("b", arrow::uint32())},
                                        }),
                          {R"([
                                      {"a": null, "b": 5},
                                      {"a": 1,    "b": 3},
                                      {"a": 3,    "b": null},
                                      {"a": 2,    "b": 5},
                                      {"a": 2,    "b": 6},
                                      {"a": 1,    "b": 5}
                                    ])"});
  std::shared_ptr<Table> table, out;
  CHECK_CYLON_STATUS(Table::FromArrowTable(ctx, ab, table));

  SECTION("filter table - range index") {
    auto expected = TableFromJSON(arrow::schema({
                                                    {field("b", arrow::uint32())},
                                                    {field("index", arrow::int64())}
                                                }),
                                  {R"([
                                      {"b": 5,    "index": 0},
                                      {"b": null, "index": 2},
                                      {"b": 6,    "index": 4}
                                     ])"});
    SECTION("reset index = false") {
      CHECK_CYLON_STATUS(indexing::SelectTableByRows(table, ArrayFromJSON(arrow::int64(), "[0, 2, 4]"),
                                                     {1}, &out, /*bounds_check=*/false, /*reset_index=*/false));
      CHECK_ARROW_EQUAL(expected, out->get_table());

      int64_t test_loc = -1;
      CHECK_CYLON_STATUS(out->GetArrowIndex()->LocationByValue(arrow::MakeScalar<int64_t>(2), &test_loc));
      REQUIRE(test_loc == 1);
    }

    SECTION("reset index = true") {
      CHECK_CYLON_STATUS(indexing::SelectTableByRows(table, ArrayFromJSON(arrow::int64(), "[0, 2, 4]"),
                                                     {1}, &out, /*bounds_check=*/false, /*reset_index=*/true));
      CHECK_ARROW_EQUAL(expected->SelectColumns({0}).ValueOrDie(), out->get_table());

      int64_t test_loc = -1;
      CHECK_CYLON_STATUS(out->GetArrowIndex()->LocationByValue(arrow::MakeScalar<int64_t>(2), &test_loc));
      REQUIRE(test_loc == 2);
    }
  }

  for (auto index_type: {IndexingType::Linear, IndexingType::Hash}) {
    SECTION("filter table - " + std::to_string(index_type)) {
      auto expected = TableFromJSON(arrow::schema({
                                                      {field("a", arrow::uint32())},
                                                      {field("b", arrow::uint32())},
                                                  }),
                                    {R"([
                                      {"a": null, "b": 5},
                                      {"a": 3,    "b": null},
                                      {"a": 2,    "b": 6}
                                     ])"});
      // set index by column b
      CHECK_CYLON_STATUS(table->SetArrowIndex(1, index_type));

      SECTION("reset index = false") {
        // filter column a. this will add col b to the table as it is the index
        CHECK_CYLON_STATUS(indexing::SelectTableByRows(table, ArrayFromJSON(arrow::int64(), "[0, 2, 4]"),
                                                       {0}, &out, /*bounds_check=*/false, /*reset_index=*/false));
        CHECK_ARROW_EQUAL(expected, out->get_table());

        REQUIRE(out->GetArrowIndex()->GetIndexingType() == index_type);
        int64_t test_loc = -1;
        CHECK_CYLON_STATUS(out->GetArrowIndex()->LocationByValue(arrow::MakeScalar<uint32_t>(5), &test_loc));
        REQUIRE(test_loc == 0);
        CHECK_CYLON_STATUS(out->GetArrowIndex()->LocationByValue(arrow::MakeNullScalar(arrow::uint32()), &test_loc));
        REQUIRE(test_loc == 1);
      }

      SECTION("reset index = true") {
        CHECK_CYLON_STATUS(indexing::SelectTableByRows(table, ArrayFromJSON(arrow::int64(), "[0, 2, 4]"),
                                                       {0}, &out, /*bounds_check=*/false, /*reset_index=*/true));
        CHECK_ARROW_EQUAL(expected->SelectColumns({0}).ValueOrDie(), out->get_table());

        REQUIRE(out->GetArrowIndex()->GetIndexingType() == Range);
        int64_t test_loc = -1;
        CHECK_CYLON_STATUS(out->GetArrowIndex()->LocationByValue(arrow::MakeScalar<uint32_t>(2), &test_loc));
        REQUIRE(test_loc == 2);
      }
    }
  }
}

TEST_CASE("Test MaskTable", "[indexing]") {
  auto ab = TableFromJSON(arrow::schema({
                                            {field("a", arrow::uint32())},
                                            {field("b", arrow::uint32())},
                                        }),
                          {R"([
                                      {"a": null, "b": 5},
                                      {"a": 1,    "b": 3},
                                      {"a": 3,    "b": 4}
                                    ])"});

  auto mask = TableFromJSON(arrow::schema({
                                              {field("a", arrow::boolean())},
                                              {field("b", arrow::boolean())},
                                          }),
                            {R"([
                                      {"a": true,   "b": false},
                                      {"a": true,   "b": true},
                                      {"a": false,  "b": false}
                                    ])"});

  auto expected = TableFromJSON(arrow::schema({
                                                  {field("a", arrow::uint32())},
                                                  {field("b", arrow::uint32())},
                                              }),
                                {R"([
                                      {"a": null,   "b": null},
                                      {"a": 1,      "b": 3},
                                      {"a": null,   "b": null}
                                    ])"});

  std::shared_ptr<Table> table, cmask, out;

  SECTION("no offset") {
    CHECK_CYLON_STATUS(Table::FromArrowTable(ctx, ab, table));
    CHECK_CYLON_STATUS(Table::FromArrowTable(ctx, mask, cmask));

    CHECK_CYLON_STATUS(indexing::MaskTable(table, cmask, &out));
    CHECK_ARROW_EQUAL(expected, out->get_table());
  }

  SECTION("with offset"){
    CHECK_CYLON_STATUS(Table::FromArrowTable(ctx, ab->Slice(1, 2), table));
    CHECK_CYLON_STATUS(Table::FromArrowTable(ctx, mask->Slice(1, 2), cmask));

    CHECK_CYLON_STATUS(indexing::MaskTable(table, cmask, &out));
    CHECK_ARROW_EQUAL(expected->Slice(1, 2), out->get_table());
  }
}

} // namespace test
} // namespace cylon

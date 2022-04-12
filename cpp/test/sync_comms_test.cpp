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
#include <arrow/visitor_inline.h>

#include "common/test_header.hpp"

namespace cylon {
namespace test {

enum GenType { Empty, Null, NonEmpty };

void generate_table(std::shared_ptr<arrow::Schema> *schema,
                    std::shared_ptr<arrow::Table> *table, GenType type = NonEmpty) {
  *schema = arrow::schema({
                              {field("_", arrow::boolean())},
                              {field("a", arrow::uint32())},
                              {field("b", arrow::float64())},
                              {field("c", arrow::utf8())},
                          });
  switch (type) {
    case Empty:*table = TableFromJSON(*schema, {R"([])"});
      return;
    case Null:
      *table = TableFromJSON(*schema, {R"([{"_": null, "a": null, "b": null, "c": null}])"});
      return;
    case NonEmpty:
      *table = TableFromJSON(*schema, {R"([{"_": true,  "a": null, "b": 5,  "c": "1"},
                                     {"_": false,  "a": 1,    "b": 3,    "c": "12"},
                                     {"_": true,  "a": 3,    "b": null, "c": "123"},
                                     {"_": null,  "a": null, "b": null, "c": null},
                                     {"_": true,  "a": 2,    "b": 5,    "c": "1234"},
                                     {"_": false,  "a": 1,    "b": 5,    "c": null}
                                    ])"});
      return;
  }
}

TEST_CASE("all gather table", "[sync comms]") {
  std::shared_ptr<arrow::Schema> schema;
  std::shared_ptr<arrow::Table> in_table;
  generate_table(&schema, &in_table);

  auto test_all_gather = [](const auto &atable) {
    auto table = std::make_shared<Table>(ctx, atable);

    std::vector<std::shared_ptr<Table>> out;

    const auto &comm = ctx->GetCommunicator();
    CHECK_CYLON_STATUS(comm->AllGather(table, &out));
    REQUIRE((int) out.size() == WORLD_SZ);

    INFO ("world sz " + std::to_string(WORLD_SZ) + " rank " + std::to_string(RANK));
    for (int i = 0; i < WORLD_SZ; i++) {
      CHECK_ARROW_EQUAL(atable, out[i]->get_table());
    }
  };

  for (const auto &atable: {in_table, in_table->Slice(3)}) {
    SECTION(atable.get() == in_table.get() ? "without offset" : "with offset") {
      test_all_gather(atable);
    }
  }

  SECTION("all empty") {
    generate_table(&schema, &in_table, Empty);
    test_all_gather(in_table);
  }

  SECTION("all null with single line") {
    generate_table(&schema, &in_table, Null);
    test_all_gather(in_table);
  }

  SECTION("some empty") {
    std::shared_ptr<arrow::Table> empty_table;
    generate_table(&schema, &in_table, NonEmpty);
    generate_table(&schema, &empty_table, Empty);

    // make even ranked tables empty
    std::shared_ptr<Table> table;
    if (RANK % 2 == 0) {
      table = std::make_shared<Table>(ctx, empty_table);
    } else {
      table = std::make_shared<Table>(ctx, in_table);
    }

    std::vector<std::shared_ptr<Table>> out;

    const auto &comm = ctx->GetCommunicator();
    CHECK_CYLON_STATUS(comm->AllGather(table, &out));
    REQUIRE((int) out.size() == WORLD_SZ);

    INFO ("world sz " + std::to_string(WORLD_SZ) + " rank " + std::to_string(RANK));
    for (int i = 0; i < WORLD_SZ; i++) {
      if (i % 2 == 0) {
        CHECK_ARROW_EQUAL(empty_table, out[i]->get_table());
      } else {
        CHECK_ARROW_EQUAL(in_table, out[i]->get_table());
      }
    }
  }
}

TEST_CASE("gather table", "[sync comms]") {
  std::shared_ptr<arrow::Schema> schema;
  std::shared_ptr<arrow::Table> in_table;
  generate_table(&schema, &in_table);

  int gather_root = WORLD_SZ / 2;

  auto test_gather = [&](const auto &atable) {
    auto table = std::make_shared<Table>(ctx, atable);

    std::vector<std::shared_ptr<Table>> out;

    const auto &comm = ctx->GetCommunicator();
    CHECK_CYLON_STATUS(comm->Gather(table, gather_root, true, &out));

    if (gather_root == RANK) {
      REQUIRE((int) out.size() == WORLD_SZ);

      INFO ("world sz " + std::to_string(WORLD_SZ) + " rank " + std::to_string(RANK));
      for (int i = 0; i < WORLD_SZ; i++) {
        INFO("Checking " + std::to_string(i) + " table")
        CHECK_ARROW_EQUAL(atable, out[i]->get_table());
      }
    } else {
      REQUIRE(out.empty());
    }
  };

  for (const auto &atable: {in_table, in_table->Slice(3)}) {
    SECTION(atable.get() == in_table.get() ? "without offset" : "with offset") {
      test_gather(atable);
    }
  }

  SECTION("all empty") {
    generate_table(&schema, &in_table, Empty);
    test_gather(in_table);
  }

  SECTION("all null with single line") {
    generate_table(&schema, &in_table, Null);
    test_gather(in_table);
  }

  SECTION("some empty") {
    std::shared_ptr<arrow::Table> empty_table;
    generate_table(&schema, &in_table, NonEmpty);
    generate_table(&schema, &empty_table, Empty);

    // make even ranked tables empty
    std::shared_ptr<Table> table;
    if (RANK % 2 == 0) {
      table = std::make_shared<Table>(ctx, empty_table);
    } else {
      table = std::make_shared<Table>(ctx, in_table);
    }

    std::vector<std::shared_ptr<Table>> out;
    const auto &comm = ctx->GetCommunicator();
    CHECK_CYLON_STATUS(comm->Gather(table, gather_root, true, &out));

    if (gather_root == RANK) {
      REQUIRE((int) out.size() == WORLD_SZ);

      INFO ("world sz " + std::to_string(WORLD_SZ) + " rank " + std::to_string(RANK));
      for (int i = 0; i < WORLD_SZ; i++) {
        if (i % 2 == 0) {
          CHECK_ARROW_EQUAL(empty_table, out[i]->get_table());
        } else {
          CHECK_ARROW_EQUAL(in_table, out[i]->get_table());
        }
      }
    } else {
      REQUIRE(out.empty());
    }
  }
}

TEST_CASE("bcast table", "[sync comms]") {
  std::shared_ptr<arrow::Schema> schema;
  std::shared_ptr<arrow::Table> in_table;
  generate_table(&schema, &in_table);

  int bcast_root = WORLD_SZ / 2;

  auto test_bcast = [&](const auto &atable) {
    std::shared_ptr<Table> table;
    if (bcast_root == RANK) {
      table = std::make_shared<Table>(ctx, atable);
    }

    const auto &comm = ctx->GetCommunicator();
    CHECK_CYLON_STATUS(comm->Bcast(&table, bcast_root));

    INFO ("world sz " + std::to_string(WORLD_SZ) + " rank " + std::to_string(RANK));
    REQUIRE(table != nullptr);
    CHECK_ARROW_EQUAL(atable, table->get_table());
  };

  for (const auto &atable: {in_table, in_table->Slice(3)}) {
    SECTION(atable.get() == in_table.get() ? "without offset" : "with offset") {
      test_bcast(atable);
    }
  }

  SECTION("empty") {
    generate_table(&schema, &in_table, Empty);
    test_bcast(in_table);
  }

  SECTION("null with single line") {
    generate_table(&schema, &in_table, Null);
    test_bcast(in_table);
  }
}

TEMPLATE_LIST_TEST_CASE("allreduce array", "[sync comms]", ArrowNumericTypes) {
  auto type = default_type_instance<TestType>();
  auto rank = *arrow::MakeScalar(RANK)->CastTo(type);
  auto base_arr =  ArrayFromJSON(type, "[1, 2, 3, 4]");

  // [1, 2, 3, 4] * rank
  auto arr = arrow::compute::Multiply(base_arr, rank)->make_array();
  auto col = Column::Make(std::move(arr));

  const auto &comm = ctx->GetCommunicator();

  auto test_allreduce = [&](net::ReduceOp op, const auto &exp) {
    std::shared_ptr<Column> res;
    CHECK_CYLON_STATUS(comm->AllReduce(col, op, &res));

    const auto &rcv = res->data();
    CHECK_ARROW_EQUAL(exp, rcv);
  };

  SECTION("sum") {
    auto multiplier = *arrow::MakeScalar((WORLD_SZ - 1) * WORLD_SZ / 2)->CastTo(type);
    auto exp = arrow::compute::Multiply(base_arr, multiplier)->make_array();
    test_allreduce(net::SUM, exp);
  }

  SECTION("min") {
    test_allreduce(net::MIN, ArrayFromJSON(type, "[0, 0, 0, 0]"));
  }

  SECTION("max") {
    auto multiplier = *arrow::MakeScalar(WORLD_SZ - 1)->CastTo(type);
    auto exp = arrow::compute::Multiply(base_arr, multiplier)->make_array();
    test_allreduce(net::MAX, exp);
  }
}

TEMPLATE_LIST_TEST_CASE("allgather array - numeric", "[sync comms]", ArrowNumericTypes) {
  auto type = default_type_instance<TestType>();
  INFO("type: " + type->ToString());
  const auto &comm = ctx->GetCommunicator();

  auto test_all_gather = [&](const std::shared_ptr<arrow::Array> &a_col) {
    auto c_col = Column::Make(a_col);
    std::vector<std::shared_ptr<Column>> res;
    CHECK_CYLON_STATUS(comm->Allgather(c_col, &res));

    for (const auto &r: res) {
      CHECK_ARROW_EQUAL(a_col, r->data());
    }
  };

  SECTION("no nulls") {
    const auto &arr = ArrayFromJSON(type,
                                    "[111, 112, 113, 114, 115, 116, 117, 118, 119, 110, 111, 121]");
    for (const auto &col: {arr, arr->Slice(3)}) {
      SECTION(col.get() == arr.get() ? "without offset" : "with offset") {
        test_all_gather(col);
      }
    }
  }

  SECTION("with nulls") {
    const auto &arr = ArrayFromJSON(type, "[1, 2, 3, null, 5, 6, 7, 8, 9, 10, null, 12]");
    for (const auto &col: {arr, arr->Slice(3)}) {
      SECTION(col.get() == arr.get() ? "without offset" : "with offset") {
        test_all_gather(col);
      }
    }
  }
}

TEMPLATE_LIST_TEST_CASE("allgather array - binary", "[sync comms]", ArrowBinaryTypes) {
  auto type = default_type_instance<TestType>();
  INFO("type: " + type->ToString());
  const auto &comm = ctx->GetCommunicator();

  auto test_all_gather = [&](const std::shared_ptr<arrow::Array> &a_col) {
    auto c_col = Column::Make(a_col);
    std::vector<std::shared_ptr<Column>> res;
    CHECK_CYLON_STATUS(comm->Allgather(c_col, &res));

    for (const auto &r: res) {
      CHECK_ARROW_EQUAL(a_col, r->data());
    }
  };

  SECTION("no nulls") {
    const auto &arr = ArrayFromJSON(type,
                                    R"(["111", "112", "113", "114", "115", "116",
                                    "117", "118", "119", "110", "111", "121"])");
    for (const auto &col: {arr, arr->Slice(3)}) {
      SECTION(col.get() == arr.get() ? "without offset" : "with offset") {
        test_all_gather(col);
      }
    }
  }

  SECTION("with nulls") {
    const auto &arr =
        ArrayFromJSON(type, R"(["1", "2", "3", null, "5", "6", "7", "8", "9", "10", null, "12"])");
    for (const auto &col: {arr, arr->Slice(3)}) {
      SECTION(col.get() == arr.get() ? "without offset" : "with offset") {
        test_all_gather(col);
      }
    }
  }
}

TEMPLATE_LIST_TEST_CASE("allgather scalar - numeric", "[sync comms]", ArrowNumericTypes) {
  auto type = default_type_instance<TestType>();
  INFO("type: " + type->ToString());
  const auto &comm = ctx->GetCommunicator();

  auto test_all_gather = [&](const std::shared_ptr<arrow::Scalar> &s_) {
    auto c_col = Scalar::Make(s_);
    std::shared_ptr<Column> res;
    CHECK_CYLON_STATUS(comm->Allgather(c_col, &res));

    auto exp = arrow::MakeArrayFromScalar(*s_, WORLD_SZ).ValueOrDie();
    CHECK_ARROW_EQUAL(exp, res->data());
  };

  SECTION("no nulls") {
    auto s = arrow::MakeScalar(type, 100).ValueOrDie();
    test_all_gather(s);
  }

  SECTION("with nulls") {
    auto s = arrow::MakeNullScalar(type);
    test_all_gather(s);
  }
}

TEMPLATE_LIST_TEST_CASE("allgather scalar - binary", "[sync comms]", ArrowBinaryTypes) {
  auto type = default_type_instance<TestType>();
  INFO("type: " + type->ToString());
  const auto &comm = ctx->GetCommunicator();

  auto test_all_gather = [&](const std::shared_ptr<arrow::Scalar> &s_) {
    auto c_col = Scalar::Make(s_);
    std::shared_ptr<Column> res;
    CHECK_CYLON_STATUS(comm->Allgather(c_col, &res));

    auto exp = arrow::MakeArrayFromScalar(*s_, WORLD_SZ).ValueOrDie();
    CHECK_ARROW_EQUAL(exp, res->data());
  };

  SECTION("no nulls") {
    auto s = MakeBinaryScalar<TestType>("aaaa");
    test_all_gather(s);
  }

  SECTION("with nulls") {
    auto s = arrow::MakeNullScalar(type);
    test_all_gather(s);
  }
}

}
}
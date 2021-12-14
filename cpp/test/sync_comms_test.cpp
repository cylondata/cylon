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

    const auto &sync_comm = ctx->sync_communicator();
    CHECK_CYLON_STATUS(sync_comm->AllGather(table, &out));
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

    const auto &sync_comm = ctx->sync_communicator();
    CHECK_CYLON_STATUS(sync_comm->AllGather(table, &out));
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

    const auto &sync_comm = ctx->sync_communicator();
    CHECK_CYLON_STATUS(sync_comm->Gather(table, gather_root, true, &out));

    if (gather_root == RANK) {
      REQUIRE((int) out.size() == WORLD_SZ);

      INFO ("world sz " + std::to_string(WORLD_SZ) + " rank " + std::to_string(RANK));
      for (int i = 0; i < WORLD_SZ; i++) {
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
    const auto &sync_comm = ctx->sync_communicator();
    CHECK_CYLON_STATUS(sync_comm->Gather(table, gather_root, true, &out));

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

}
}
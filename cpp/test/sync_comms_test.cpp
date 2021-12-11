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

TEST_CASE("all gather table", "[sync comms]") {
  auto schema = arrow::schema({
                                  {field("_", arrow::boolean())},
                                  {field("a", arrow::uint32())},
                                  {field("b", arrow::float64())},
                                  {field("c", arrow::utf8())},
                              });

  auto in_table = TableFromJSON(schema, {R"([{"_": true,  "a": null, "b": 5,  "c": "1"},
                                     {"_": false,  "a": 1,    "b": 3,    "c": "12"},
                                     {"_": true,  "a": 3,    "b": null, "c": "123"},
                                     {"_": null,  "a": null, "b": null, "c": null},
                                     {"_": true,  "a": 2,    "b": 5,    "c": "1234"},
                                     {"_": false,  "a": 1,    "b": 5,    "c": null}
                                    ])"});
  auto table = std::make_shared<Table>(ctx, in_table);

  std::vector<std::shared_ptr<Table>> out;

  const auto &sync_comm = ctx->sync_communicator();
  CHECK_CYLON_STATUS(sync_comm->AllGather(table, &out));
  REQUIRE((int) out.size() == WORLD_SZ);

  INFO ("world sz " + std::to_string(WORLD_SZ) + " rank " + std::to_string(RANK));
  for (int i = 0; i < WORLD_SZ; i++) {
    CHECK_ARROW_EQUAL(in_table, out[i]->get_table());
  }
}

TEST_CASE("gather table", "[sync comms]") {
  auto schema = arrow::schema({
                                  {field("_", arrow::boolean())},
                                  {field("a", arrow::uint32())},
                                  {field("b", arrow::float64())},
                                  {field("c", arrow::utf8())},
                              });

  auto in_table = TableFromJSON(schema, {R"([{"_": true,  "a": null, "b": 5,  "c": "1"},
                                     {"_": false,  "a": 1,    "b": 3,    "c": "12"},
                                     {"_": true,  "a": 3,    "b": null, "c": "123"},
                                     {"_": null,  "a": null, "b": null, "c": null},
                                     {"_": true,  "a": 2,    "b": 5,    "c": "1234"},
                                     {"_": false,  "a": 1,    "b": 5,    "c": null}
                                    ])"});
  auto table = std::make_shared<Table>(ctx, in_table);
  int gather_root = WORLD_SZ/2;

  std::vector<std::shared_ptr<Table>> out;

  const auto &sync_comm = ctx->sync_communicator();
  CHECK_CYLON_STATUS(sync_comm->Gather(table, gather_root, true, &out));

  if (gather_root == RANK){
    REQUIRE((int) out.size() == WORLD_SZ);

    INFO ("world sz " + std::to_string(WORLD_SZ) + " rank " + std::to_string(RANK));
    for (int i = 0; i < WORLD_SZ; i++) {
      CHECK_ARROW_EQUAL(in_table, out[i]->get_table());
    }
  } else {
    REQUIRE(out.empty());
  }
}

}
}
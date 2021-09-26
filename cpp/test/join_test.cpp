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
#include "test_utils.hpp"

namespace cylon {
namespace test {

TEST_CASE("Join testing", "[join]") {
  std::string path1 = "../data/input/csv1_" + std::to_string(RANK) + ".csv";
  std::string path2 = "../data/input/csv2_" + std::to_string(RANK) + ".csv";
  std::string out_path =
      "../data/output/join_inner_" + std::to_string(WORLD_SZ) + "_" + std::to_string(RANK) + ".csv";

  SECTION("testing inner joins - sort") {
    const auto &join_config = join::config::JoinConfig::InnerJoin(0, 0, cylon::join::config::JoinAlgorithm::SORT);
    test::TestJoinOperation(join_config, ctx, path1, path2, out_path);
  }

  SECTION("testing inner joins - hash") {
    const auto &join_config = join::config::JoinConfig::InnerJoin(0, 0, cylon::join::config::JoinAlgorithm::HASH);
    test::TestJoinOperation(join_config, ctx, path1, path2, out_path);
  }
}

TEST_CASE("Multi Index Join testing", "[multi_join]") {
  std::string path1 = "../data/input/multi_join1.csv";
  std::string path2 = "../data/input/multi_join2.csv";
  std::string out_path = "../data/output/multi_join_" + std::to_string(WORLD_SZ) + "_" + std::to_string(RANK) + ".csv";

  SECTION("testing inner joins - sort") {
    const auto &jc =
        join::config::JoinConfig::InnerJoin({0, 1}, {0, 1}, cylon::join::config::JoinAlgorithm::SORT, "l_", "r_");
    test::TestJoinOperation(jc, ctx, path1, path2, out_path);
  }

  SECTION("testing inner joins - hash") {
    const auto &jc =
        join::config::JoinConfig::InnerJoin({0, 1}, {0, 1}, cylon::join::config::JoinAlgorithm::HASH, "l_", "r_");
    test::TestJoinOperation(jc, ctx, path1, path2, out_path);
  }
}

TEST_CASE("Join testing chunks", "[join]") {
  std::shared_ptr<Table> _t1, _t2, t1, t2, out;
  std::shared_ptr<arrow::Table> at1, at2;

  CHECK_CYLON_STATUS(test::CreateTable(ctx, 4, _t1)); // create dummy table
  CHECK_CYLON_STATUS(_t1->ToArrowTable(at1)); // convert to arrow
  auto res1 = arrow::ConcatenateTables({at1, at1}); // concat same table
  REQUIRE(res1.ok());
  CHECK_CYLON_STATUS(Table::FromArrowTable(ctx, res1.ValueOrDie(), t1)); // use concat table

  CHECK_CYLON_STATUS(test::CreateTable(ctx, 2, _t2)); // create dummy table
  CHECK_CYLON_STATUS(_t2->ToArrowTable(at2)); // convert to arrow
  auto res2 = arrow::ConcatenateTables({at2, at2}); // concat same table
  REQUIRE(res2.ok());
  CHECK_CYLON_STATUS(Table::FromArrowTable(ctx, res2.ValueOrDie(), t2)); // use concat table

  SECTION("testing inner joins - sort") {
    const auto &jc = join::config::JoinConfig::InnerJoin(0, 0, join::config::JoinAlgorithm::SORT);
    CHECK_CYLON_STATUS(DistributedJoin(t1, t2, jc, out)); // just check if runs without a problem
  }

  SECTION("testing inner joins - hash") {
    const auto &jc = join::config::JoinConfig::InnerJoin(0, 0, join::config::JoinAlgorithm::HASH);
    CHECK_CYLON_STATUS(DistributedJoin(t1, t2, jc, out)); // just check if runs without a problem
  }

  SECTION("testing inner joins - sort") {
    const auto &jc = join::config::JoinConfig::InnerJoin({0, 1}, {0, 1}, join::config::JoinAlgorithm::SORT);
    CHECK_CYLON_STATUS(DistributedJoin(t1, t2, jc, out)); // just check if runs without a problem
  }

  SECTION("testing inner joins - hash") {
    const auto &jc = join::config::JoinConfig::InnerJoin({0, 1}, {0, 1}, join::config::JoinAlgorithm::HASH);
    CHECK_CYLON_STATUS(DistributedJoin(t1, t2, jc, out)); // just check if runs without a problem
  }
}

}
}
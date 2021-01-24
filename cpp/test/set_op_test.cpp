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

#include "test_utils.hpp"
#include "test_header.hpp"

#include "compute/aggregates.hpp"

using namespace cylon;

TEST_CASE("Set operation testing", "[set_op]") {
  std::string path1 = "../data/input/csv1_" + std::to_string(RANK) + ".csv";
  std::string path2 = "../data/input/csv2_" + std::to_string(RANK) + ".csv";
  std::string out_path;

  SECTION("testing union") {
    out_path =
        "../data/output/union_" + std::to_string(WORLD_SZ) + "_" + std::to_string(RANK) + ".csv";
    REQUIRE(test::TestSetOperation(&DistributedUnion, ctx, path1, path2, out_path) == 0);
  }

  SECTION("testing union itself") {
    std::shared_ptr<Table> t, local, dist;
    auto s = FromCSV(ctx, "../data/input/csv1_0.csv", t);
    REQUIRE(s.is_ok());

    s = Union(t, t, local);
    REQUIRE(s.is_ok());

    s = DistributedUnion(t, t, dist);
    REQUIRE(s.is_ok());

    std::shared_ptr<compute::Result> res;
    s = compute::Count(dist, 0, res);
    REQUIRE((s.is_ok()
        && std::static_pointer_cast<arrow::Int64Scalar>(res->GetResult().scalar())->value == local->Rows()));
  }

  SECTION("testing intersection itself") {
    std::shared_ptr<Table> t, local, dist;
    auto s = FromCSV(ctx, "../data/input/csv1_0.csv", t);
    REQUIRE(s.is_ok());

    s = Intersect(t, t, local);
    REQUIRE(s.is_ok());

    s = DistributedIntersect(t, t, dist);
    REQUIRE(s.is_ok());

    std::shared_ptr<compute::Result> res;
    s = compute::Count(dist, 0, res);
    REQUIRE((s.is_ok()
        && std::static_pointer_cast<arrow::Int64Scalar>(res->GetResult().scalar())->value == local->Rows()));
  }

  SECTION("testing subtract") {
    out_path =
        "../data/output/subtract_" + std::to_string(WORLD_SZ) + "_" + std::to_string(RANK) + ".csv";
    REQUIRE(test::TestSetOperation(&DistributedSubtract, ctx, path1, path2, out_path) == 0);
  }

  SECTION("testing subtract") {
    out_path =
        "../data/output/intersect_" + std::to_string(WORLD_SZ) + "_" + std::to_string(RANK)
            + ".csv";
    REQUIRE(test::TestSetOperation(&DistributedIntersect, ctx, path1, path2, out_path) == 0);
  }
}

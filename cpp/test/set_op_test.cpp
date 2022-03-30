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
#include "test_macros.hpp"

#include <cylon/compute/aggregates.hpp>

namespace cylon {
namespace test {

void global_sum_equal(int64_t exp, int64_t val) {
  auto rows = Scalar::Make(arrow::MakeScalar(val));
  std::shared_ptr<Scalar> res;
  CHECK_CYLON_STATUS(ctx->GetCommunicator()->AllReduce(rows, net::SUM, &res));
  CHECK_ARROW_EQUAL(arrow::MakeScalar(exp), res->data());
}

TEST_CASE("Set operation testing", "[set_op]") {
  std::string path1 = "../data/input/csv1_" + std::to_string(RANK) + ".csv";
  std::string path2 = "../data/input/csv2_" + std::to_string(RANK) + ".csv";
  std::string out_path;

  SECTION("testing union") {
    out_path =
        "../data/output/union_" + std::to_string(WORLD_SZ) + "_" + std::to_string(RANK) + ".csv";
    test::TestSetOperation(&DistributedUnion, ctx, path1, path2, out_path);
  }

  SECTION("testing union itself") {
    std::shared_ptr<Table> t, local, dist;
    CHECK_CYLON_STATUS(FromCSV(ctx, "../data/input/csv1_0.csv", t));
    CHECK_CYLON_STATUS(Union(t, t, local));
    CHECK_CYLON_STATUS(DistributedUnion(t, t, dist));

    global_sum_equal(local->Rows(), dist->Rows());
  }

  SECTION("testing intersection itself") {
    std::shared_ptr<Table> t, local, dist;
    CHECK_CYLON_STATUS(FromCSV(ctx, "../data/input/csv1_0.csv", t));
    CHECK_CYLON_STATUS(Intersect(t, t, local));
    CHECK_CYLON_STATUS(DistributedIntersect(t, t, dist));

    global_sum_equal(local->Rows(), dist->Rows());
  }

  SECTION("testing subtract") {
    out_path = "../data/output/subtract_" + std::to_string(WORLD_SZ) + "_" + std::to_string(RANK) + ".csv";
    TestSetOperation(&DistributedSubtract, ctx, path1, path2, out_path);
  }

  SECTION("testing intersect") {
    out_path = "../data/output/intersect_" + std::to_string(WORLD_SZ) + "_" + std::to_string(RANK) + ".csv";
    TestSetOperation(&DistributedIntersect, ctx, path1, path2, out_path);
  }
}

}
}

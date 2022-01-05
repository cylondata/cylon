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

TEST_CASE("Parquet join testing", "[join]") {
  std::string path1 = "../data/input/parquet1_" + std::to_string(RANK) + ".parquet";
  std::string path2 = "../data/input/parquet2_" + std::to_string(RANK) + ".parquet";
  std::string out_path =
      "../data/output/join_inner_" + std::to_string(WORLD_SZ) + "_" + std::to_string(RANK) + ".parquet";

  SECTION("testing inner joins") {
    auto join_config = join::config::JoinConfig(join::config::JoinType::INNER, 0, 0,
                                                join::config::JoinAlgorithm::SORT,
                                                "lt-", "rt-");
    TestParquetJoinOperation(join_config, ctx, path1, path2, out_path);
  }
}

}
}

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

#include <cylon/indexing/index.hpp>
#include "common/test_header.hpp"
#include "test_index_utils.hpp"

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
    for (auto &item: out_files) {
      TestIndexBuildOperation(ctx, path1, item.first);
    }
  }

  SECTION("testing loc index 1") {
    for (auto &item: out_files) {
      TestIndexLocOperation1(ctx, path1, item.first, item.second);
    }
  }

  SECTION("testing loc index 2") {
    for (auto &item: out_files) {
      TestIndexLocOperation2(ctx, path1, item.first, item.second);
    }
  }

  SECTION("testing loc index 3") {
    for (auto &item: out_files) {
      TestIndexLocOperation3(ctx, path1, item.first, item.second);
    }
  }

  SECTION("testing loc index 4") {
    for (auto &item: out_files) {
      TestIndexLocOperation4(ctx, path1, item.first, item.second);
    }
  }

  SECTION("testing loc index 5") {
    for (auto &item: out_files) {
      TestIndexLocOperation5(ctx, path1, item.first, item.second);
    }
  }

  SECTION("testing loc index 6") {
    for (auto &item: out_files) {
      TestIndexLocOperation6(ctx, path1, item.first, item.second);
    }
  }
}

} // namespace test
} // namespace cylon

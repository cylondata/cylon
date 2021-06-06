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

#include "test_header.hpp"

#include "indexing/index.hpp"
#include "test_index_utils.hpp"

using namespace cylon;

TEST_CASE("Index testing", "[indexing]") {
std::string path1 = "../data/input/indexing_data.csv";

std::vector<std::string> output_files {"../data/output/indexing_loc_hl_",
                                       "../data/output/indexing_loc_hl_",
                                       "../data/output/indexing_loc_r_"};


std::vector<cylon::IndexingType> indexing_types = {cylon::IndexingType::Hash,
                                                       cylon::IndexingType::Linear,
                                                       cylon::IndexingType::Range};

SECTION("testing build index") {
  for(auto type : indexing_types) {
    REQUIRE(test::TestIndexBuildOperation(path1, type) == 0);
  }
}
SECTION("testing loc index 1") {
for(size_t i=0; i < output_files.size(); i++) {
  auto type = indexing_types.at(i);
  auto output_file = output_files.at(i);
  REQUIRE(test::TestIndexLocOperation1(path1, type, output_file) == 0);
}

SECTION("testing loc index 2") {
for(size_t i = 0; i<output_files.size(); i++) {
  auto type = indexing_types.at(i);
  auto output_file = output_files.at(i);
  REQUIRE(test::TestIndexLocOperation2(path1, type, output_file) == 0);
}
}

SECTION("testing loc index 3") {
  for(size_t i = 0; i<output_files.size(); i++) {
    auto type = indexing_types.at(i);
    auto output_file = output_files.at(i);
    REQUIRE(test::TestIndexLocOperation3(path1, type, output_file) == 0);
  }
}

SECTION("testing loc index 4") {
  for(size_t i = 0; i<output_files.size(); i++) {
    auto type = indexing_types.at(i);
    auto output_file = output_files.at(i);
    REQUIRE(test::TestIndexLocOperation4(path1, type, output_file) == 0);
  }
}

SECTION("testing loc index 5") {
  for(size_t i = 0; i<output_files.size(); i++) {
    auto type = indexing_types.at(i);
    auto output_file = output_files.at(i);
    REQUIRE(test::TestIndexLocOperation5(path1, type, output_file) == 0);
  }
}

SECTION("testing loc index 6") {
  for(size_t i = 0; i<output_files.size(); i++) {
    auto type = indexing_types.at(i);
    auto output_file = output_files.at(i);
    REQUIRE(test::TestIndexLocOperation6(path1, type, output_file) == 0);
  }
}

}

}

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
#include "test_utils.hpp"
#include "indexing/index.hpp"

using namespace cylon;

TEST_CASE("Index testing", "[indexing]") {
std::string path1 = "../data/input/indexing_data.csv";
std::string out_path = "../data/output/indexing_config_1.csv";
std::vector<cylon::IndexingSchema> indexing_schemas = {cylon::IndexingSchema::Hash,
                                                       cylon::IndexingSchema::Linear,
                                                       cylon::IndexingSchema::Range};

SECTION("testing build index") {
for(auto schema : indexing_schemas) {
REQUIRE(test::TestIndexBuildOperation(path1, schema) == 0);
}
}
SECTION("testing loc index 1") {
for(auto schema : indexing_schemas) {
REQUIRE(test::TestIndexLocOperation1(path1, schema) == 0);
}

}
}

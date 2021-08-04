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

#include <examples/gcylon/print.hpp>
#include "common/test_header.hpp"
#include "gcylon/test_gutils.hpp"
#include <gcylon/utils/util.hpp>
#include <gcylon/utils/construct.hpp>

using namespace cylon;
using namespace gcylon;

TEST_CASE("shuffling cudf tables", "[gshuffle]") {

  SECTION("testing int64 column based shuffling") {
      std::string inputFileName = "../../data/input/csv1_" + std::to_string(RANK) + ".csv";
      std::string outputFileName = "../../data/output/shuffled_" + std::to_string(RANK) + ".csv";

      std::ifstream infile(inputFileName);
      REQUIRE((infile.good()));

      REQUIRE((gcylon::test::PerformShuffleTest(inputFileName, outputFileName, RANK)));
  }

  SECTION("testing string column based shuffling") {
      std::string inputFileName = "../../data/input/cities_" + std::to_string(RANK) + ".csv";
      std::string outputFileName = "../../data/output/shuffled_cities_" + std::to_string(RANK) + ".csv";

      std::ifstream infile(inputFileName);
      REQUIRE((infile.good()));

      REQUIRE((gcylon::test::PerformShuffleTest(inputFileName, outputFileName, RANK)));
  }

}
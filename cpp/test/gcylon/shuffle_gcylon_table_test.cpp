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
#include "gcylon/test_gutils.hpp"
#include <gcylon/utils/util.hpp>

using namespace cylon;
using namespace gcylon;

TEST_CASE("shuffling cudf tables", "[gshuffle]") {

  SECTION("testing int64 column based shuffling") {
    std::string input_filename = "../../data/input/cities_a_" + std::to_string(RANK) + ".csv";
    std::string int_shuffle_filename = "../../data/output/shuffle_int_cities_a_" + std::to_string(RANK) + ".csv";
    std::string str_shuffle_filename = "../../data/output/shuffle_str_cities_a_" + std::to_string(RANK) + ".csv";

    std::ifstream infile(input_filename);
    REQUIRE((infile.good()));

    // population is at index 2 on the dataframe
    // perform int based shuffle
    int shuffle_index = 2;
    REQUIRE((gcylon::test::PerformShuffleTest(input_filename, int_shuffle_filename, shuffle_index)));

    // state_id is at index 1 on the dataframe
    // perform string based shuffle
    shuffle_index = 1;
    REQUIRE((gcylon::test::PerformShuffleTest(input_filename, str_shuffle_filename, shuffle_index)));
  }

}
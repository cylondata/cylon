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

TEST_CASE("Distributed Sorting CuDF tables", "[gsort]") {

  SECTION("testing Distributed Sort of CuDF tables") {

    std::vector<int> sort_roots{0, 1};
    std::vector<std::string> input_file_bases{"../../data/mpiops/numeric_",
                                              "../../data/mpiops/sales_nulls_nunascii_"};
    std::vector<std::string> sorted_file_bases{"../../data/sorting/numeric_sorted_",
                                               "../../data/sorting/sales_sorted_"};
    std::vector<std::vector<std::string>> column_name_vectors{{"0",       "1"},
                                                              {"Country", "Item Type", "Order Date", "Order ID", "Units Sold", "Unit Price"}};
    std::vector<std::vector<std::string>> date_column_vectors{{},
                                                              {"Order Date"}};
    std::vector<int32_t> sort_columns{0, 1};

    for (long unsigned int i = 0; i < input_file_bases.size(); i++) {
      std::string input_file = input_file_bases[i] + std::to_string(RANK) + ".csv";
      std::string sorted_file = sorted_file_bases[i] + std::to_string(RANK) + ".csv";
      std::ifstream in_file1(input_file);
      REQUIRE((in_file1.good()));
      std::ifstream in_file2(sorted_file);
      REQUIRE((in_file2.good()));

      for (int sort_root: sort_roots) {
        REQUIRE((gcylon::test::PerformSortTest(input_file,
                                               sorted_file,
                                               column_name_vectors[i],
                                               date_column_vectors[i],
                                               sort_root,
                                               sort_columns,
                                               ctx)));
      }
    }
  }
}

TEST_CASE("Distributed Sorting of sliced CuDF tables", "[gsort]") {

  SECTION("testing Distributed Sorting of sliced CuDF tables") {

    std::string input_file_base = "../../data/mpiops/sales_nulls_nunascii_";
    std::string sorted_file_base = "../../data/sorting/sales_sliced_sorted_";
    std::vector<std::string> column_names{"Country", "Item Type", "Order Date", "Order ID", "Units Sold", "Unit Price"};
    std::vector<std::string> date_columns{"Order Date"};
    std::vector<int32_t> sort_columns{1, 4};
    std::vector<int32_t> slice_range{8, 16};

    std::string input_file = input_file_base + std::to_string(RANK) + ".csv";
    std::string sorted_file = sorted_file_base + std::to_string(RANK) + ".csv";
    std::ifstream in_file1(input_file);
    REQUIRE((in_file1.good()));
    in_file1.close();
    std::ifstream in_file2(sorted_file);
    REQUIRE((in_file2.good()));
    in_file2.close();

    REQUIRE((gcylon::test::PerformSlicedSortTest(input_file,
                                                 sorted_file,
                                                 column_names,
                                                 date_columns,
                                                 sort_columns,
                                                 slice_range,
                                                 ctx)));
  }

}

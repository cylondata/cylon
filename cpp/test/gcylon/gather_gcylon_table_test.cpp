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

TEST_CASE("MPI Gather CuDF tables", "[ggather]") {

  SECTION("testing MPI gather of CuDF tables") {

    std::vector<int> gather_roots{0, 1};
    std::vector<bool> gather_from_root{false, true};
    std::vector<std::string> input_file_bases{"../../data/mpiops/numeric_",
                                              "../../data/mpiops/sales_nulls_nunascii_"};
    std::vector<std::vector<std::string>> column_name_vectors{{"0",       "1"},
                                                              {"Country", "Item Type", "Order Date", "Order ID", "Units Sold", "Unit Price"}};
    std::vector<std::vector<std::string>> date_column_vectors{{},
                                                              {"Order Date"}};

    for (long unsigned int i = 0; i < input_file_bases.size(); i++) {
      auto tables = gcylon::test::readTables(input_file_bases[i], column_name_vectors[i], date_column_vectors[i]);
      REQUIRE((tables.size() == WORLD_SZ));

      for (bool gather_root_table: gather_from_root) {
        for (int gather_root: gather_roots) {
          REQUIRE((gcylon::test::PerformGatherTest(tables, gather_root, gather_root_table, ctx)));
        }
      }
    }
  }
}

TEST_CASE("MPI Gather sliced CuDF tables", "[ggather]") {

  SECTION("testing MPI gather for sliced CuDF tables") {

    std::string input_file_base = "../../data/mpiops/sales_nulls_nunascii_";
    std::vector<std::string> column_names{"Country", "Item Type", "Order Date", "Order ID", "Units Sold", "Unit Price"};
    std::vector<std::string> date_columns{"Order Date"};

    auto tables = gcylon::test::readTables(input_file_base, column_names, date_columns);
    REQUIRE((tables.size() == WORLD_SZ));

    // make sure row ranges do not go out of range
    int COUNT = 5;
    for (auto const& tbl: tables) {
      REQUIRE((tbl->num_rows() > COUNT + 2 * COUNT));
    }

    // first check gathering empty tables
    std::vector<std::vector<int32_t>> ranges;
    for (int i = 0; i < WORLD_SZ; ++i) {
      ranges.push_back(std::vector<int32_t>{i , i});
    }
    REQUIRE((gcylon::test::PerformGatherSlicedTest(tables, ranges, ctx)));

    // first check gathering empty tables

    // check gathering single row tables
    for (int i = 0; i < COUNT; ++i) {
      std::vector<std::vector<int32_t>> ranges2{{i, i+1}, {i, i+1}, {i, i+1}, {i, i+1}};
      REQUIRE((gcylon::test::PerformGatherSlicedTest(tables, ranges2, ctx)));
    }

    // check gathering multi row tables
    for (int i = 0; i < COUNT; ++i) {
      std::vector<std::vector<int32_t>> ranges2{{i, 2 * i}, {i, 2 * i}, {i, 2 * i}, {i, 2 * i}};
      REQUIRE((gcylon::test::PerformGatherSlicedTest(tables, ranges2, ctx)));
    }
  }
}


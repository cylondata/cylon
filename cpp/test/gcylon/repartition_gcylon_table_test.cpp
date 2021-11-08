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

using namespace cylon;
using namespace gcylon;

TEST_CASE("Repartition Distributed Gcylon Table", "[grepartition]") {

  SECTION("testing Repartitioning Distributed Gcylon Table") {

    std::vector<std::vector<int>> initial_sizes{{20, 0, 0, 0},
                                                {0, 0, 0, 16},
                                                {12, 0, 0, 8},
                                                {0, 12, 20, 0},
                                                {20, 12, 16, 8}};

    std::string input_file = "../../data/mpiops/sales_nulls_nunascii_" + std::to_string(RANK) + ".csv";
    std::ifstream in_file(input_file);
    REQUIRE((in_file.good()));
    in_file.close();

    std::vector<std::string> column_names{"Country", "Item Type", "Order Date", "Order ID", "Units Sold", "Unit Price"};
    std::vector<std::string> date_columns{"Order Date"};

    // first check repartitioning evenly
    for (auto init_sizes: initial_sizes) {
      REQUIRE((gcylon::test::PerformRepartitionTest(input_file,
                                                    column_names,
                                                    date_columns,
                                                    ctx,
                                                    init_sizes)));
    }

    // second check repartitioning with given target partition sizes
    // determine target sizes randomly
    int seed = 0;
    for (auto init_sizes: initial_sizes) {
      auto all_rows = std::accumulate(init_sizes.begin(), init_sizes.end(), 0);
      auto part_sizes = gcylon::test::GenRandoms(all_rows, init_sizes.size(), seed++);
      auto all_rows2 = std::accumulate(part_sizes.begin(), part_sizes.end(), 0);
      REQUIRE((all_rows == all_rows2));
      REQUIRE((gcylon::test::PerformRepartitionTest(input_file,
                                                    column_names,
                                                    date_columns,
                                                    ctx,
                                                    init_sizes,
                                                    part_sizes)));
    }

  }


}


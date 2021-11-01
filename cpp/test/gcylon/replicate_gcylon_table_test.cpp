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

TEST_CASE("Replicate Distributed Gcylon Table", "[greplicate]") {

  SECTION("testing Replicating Distributed Gcylon Table") {

    std::string input_file_base = "../../data/mpiops/sales_nulls_nunascii_";
    std::vector<std::string> all_input_files = gcylon::test::constructInputFiles(input_file_base, WORLD_SZ);

    std::vector<std::string> column_names{"Country", "Item Type", "Order Date", "Order ID", "Units Sold", "Unit Price"};
    std::vector<std::string> date_columns{"Order Date"};

    // first check all gathering empty tables
    std::vector<std::vector<int32_t>> ranges{{0, 0}, {1, 1}, {2, 2}, {3, 3}};
    REQUIRE((gcylon::test::PerformReplicateTest(all_input_files,
                                                column_names,
                                                date_columns,
                                                ranges,
                                                ctx)));

    // check all gathering single row tables
    for (int i = 0; i < 5; ++i) {
      std::vector<std::vector<int32_t>> ranges2{{i, i+1}, {i, i+1}, {i, i+1}, {i, i+1}};
      REQUIRE((gcylon::test::PerformReplicateTest(all_input_files,
                                                  column_names,
                                                  date_columns,
                                                  ranges2,
                                                  ctx)));
    }

    // check all gathering multi row tables
    for (int i = 0; i < 5; ++i) {
      std::vector<std::vector<int32_t>> ranges2{{i, 2 * i}, {i, 2 * i}, {i, 2 * i}, {i, 2 * i}};
      REQUIRE((gcylon::test::PerformReplicateTest(all_input_files,
                                                  column_names,
                                                  date_columns,
                                                  ranges2,
                                                  ctx)));
    }

    // check all gathering full tables
    std::vector<std::vector<int32_t>> ranges3;
    ranges3.reserve(all_input_files.size());
    for (auto input_file: all_input_files) {
      cudf::io::table_with_metadata read_table = gcylon::test::readCSV(input_file, column_names, date_columns);
      std::vector<int32_t> range{0, read_table.tbl->num_rows()};
      ranges3.push_back(std::move(range));
    }
    REQUIRE((gcylon::test::PerformReplicateTest(all_input_files,
                                                column_names,
                                                date_columns,
                                                ranges3,
                                                ctx)));

  }


}


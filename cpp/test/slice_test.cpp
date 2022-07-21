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
#include <cylon/table.hpp>

namespace cylon {
namespace test {
TEST_CASE("Slice testing", "[equal]") {
    std::string path1 = "../data/input/csv1_0.csv";
    std::string path2 = "../data/input/csv1_1.csv";
    std::string path3 = "../data/input/csv1_0_shuffled.csv";
    std::string path4 = "../data/input/csv1_0_col_order_change.csv";
    std::shared_ptr<Table> table1, table2, table3, table4, out;

    auto read_options = io::config::CSVReadOptions().UseThreads(false);

    CHECK_CYLON_STATUS(FromCSV(ctx, std::vector<std::string>{path1, path2, path3, path4},
                            std::vector<std::shared_ptr<Table> *>{&table1, &table2, &table3, &table4},
                            read_options));

    SECTION("Testing Local Slice") {

        CHECK_CYLON_STATUS(LocalSlice(table1, 13, 8, out));

        CHECK_CYLON_STATUS(LocalSlice(table2, 15, 5, out));

        CHECK_CYLON_STATUS(LocalSlice(table3, 0, 10, out));

        CHECK_CYLON_STATUS(LocalSlice(table4, 2, 15, out));
    }
}

TEST_CASE("Distributed Slice testing", "[distributed slice]") {
    std::string path1 = "../data/input/csv1_" + std::to_string(RANK) +".csv";
    std::string path2 = "../data/input/csv2_" + std::to_string(RANK) +".csv";
    std::shared_ptr<Table> table1, table2, out;

    auto read_options = io::config::CSVReadOptions().UseThreads(false);

    CHECK_CYLON_STATUS(FromCSV(ctx, std::vector<std::string>{path1, path2},
                    std::vector<std::shared_ptr<Table> *>{&table1, &table2},
                            read_options));

    SECTION("Testing Distributed Slice") {
        CHECK_CYLON_STATUS(DistributedSlice(table1, 10, 15, out));

        CHECK_CYLON_STATUS(DistributedSlice(table2, 12, 8, out));
    }
}

}
}
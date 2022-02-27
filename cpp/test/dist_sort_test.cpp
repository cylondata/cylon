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
TEST_CASE("Dist sort testing", "[dist sort]") {
    std::string path1 = "../data/input/csv1_" + std::to_string(RANK) + ".csv";
    std::shared_ptr<Table> table1, table2;

    auto read_options = io::config::CSVReadOptions().UseThreads(false);

    CHECK_CYLON_STATUS(FromCSV(ctx, std::vector<std::string>{path1},
                            std::vector<std::shared_ptr<Table> *>{&table1},
                            read_options));

    SECTION("dist_sort_test_1") {
        std::shared_ptr<Table> out, out2, repartitioned;
        auto ctx = table1->GetContext();
        std::shared_ptr<arrow::Table> arrow_output;
        auto status = DistributedSortRegularSampling(table1, {0, 1}, {1, 1}, out);
        if(RANK == 0) {
        out->Print();
        }
        DistributedSort(table1, {0, 1}, out2, {1, 1});
        REQUIRE(status.is_ok());
        bool eq;
        status = DistributedEquals(out, out2, eq);
        REQUIRE(eq);
    }

    SECTION("dist_sort_test_2_different_direction") {
        std::shared_ptr<Table> out, out2, repartitioned;
        auto ctx = table1->GetContext();
        std::shared_ptr<arrow::Table> arrow_output;
        auto status = DistributedSortRegularSampling(table1, {0, 1}, {1, 0}, out);
        DistributedSort(table1, {0, 1}, out2, {1, 0});
        REQUIRE(status.is_ok());
        bool eq;
        status = DistributedEquals(out, out2, eq);
        REQUIRE(eq);
    }

    SECTION("dist_sort_test_3_different_order") {
        std::shared_ptr<Table> out, out2, repartitioned;
        auto ctx = table1->GetContext();
        std::shared_ptr<arrow::Table> arrow_output;
        auto status = DistributedSortRegularSampling(table1, {1, 0}, {0, 0}, out);
        DistributedSort(table1, {1, 0}, out2, {0, 0});
        REQUIRE(status.is_ok());
        bool eq;
        status = DistributedEquals(out, out2, eq);
        REQUIRE(eq);
    }
}

}
}
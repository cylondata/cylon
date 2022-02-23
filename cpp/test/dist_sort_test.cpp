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
    std::shared_ptr<Table> table1;

    auto read_options = io::config::CSVReadOptions().UseThreads(false);

    CHECK_CYLON_STATUS(FromCSV(ctx, std::vector<std::string>{path1},
                            std::vector<std::shared_ptr<Table> *>{&table1},
                            read_options));

    SECTION("dist_sort_test_1") {
        std::shared_ptr<Table> out;
        auto ctx = table1->GetContext();
        std::shared_ptr<arrow::Table> arrow_output;
        auto status = DistributedSortRegularSampling(table1, {0, 1}, {1, 1}, out);
        // DistributedSort(table1, {0, 1}, out2, {1, 1});
        // REQUIRE(out->Rows() == table1->Rows());
        // bool eq = true;
        // // Equals(out, out2, eq);
        // REQUIRE(out2->Rows() == table1->Rows());
        REQUIRE(status.is_ok());
        // std::cout<< RANK << std::endl;
        // out->Print();

        REQUIRE(false);
    }

    // SECTION("dist_sort_test_uniform_sample") {

    //     std::shared_ptr<Table> out;
    //     auto ctx = table1->GetContext();
    //     auto status = Sort(table1, {0, 1}, out, {1, 1});
    //     std::shared_ptr<Table> sample_result, split_points;

    //     status = SampleTableUniform(out, 15, sample_result, ctx);
    //     REQUIRE(status.is_ok());
    //     REQUIRE(sample_result->Rows() == 15);

    //     // sample_result->Print();
        
    //     status = GetSplitPoints(sample_result,
    //     {0, 1}, {1, 1}, WORLD_SZ - 1, split_points);
    //     REQUIRE(status.is_ok());
    //     REQUIRE(split_points->Rows() == WORLD_SZ - 1);

    //     std::vector<uint32_t> target_partitions, partition_hist;
    //     status = GetSplitPointIndices(split_points, out, {0, 1}, {1, 1}, target_partitions, partition_hist);
    //     REQUIRE(status.is_ok());
    // }
}

}
}
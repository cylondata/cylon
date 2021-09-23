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

TEST_CASE("MPI Broadcast a CuDF table", "[gbcast]") {

    SECTION("testing MPI broadcast of a CuDF table") {

        std::vector<int> bcast_roots{0, 1};
        std::vector<std::string> input_files {"../../data/gather/numeric_0.csv",
                                              "../../data/gather/sales_records_nulls_nunascii_0.csv"};
        std::vector<std::vector<std::string>> column_name_vectors{{"0", "1"},
                                                                 {"Country", "Item Type", "Order Date", "Order ID", "Units Sold", "Unit Price"}};
        std::vector<std::vector<std::string>> date_column_vectors{{},
                                                                  {"Order Date"}};

        for(long unsigned int i = 0; i < input_files.size(); i++) {
            std::ifstream in_file(input_files[i]);
            REQUIRE((in_file.good()));

            for(int bcast_root: bcast_roots) {
                REQUIRE((gcylon::test::PerformBcastTest(input_files[i],
                                                         column_name_vectors[i],
                                                         date_column_vectors[i],
                                                         bcast_root,
                                                         ctx)));
            }
        }
    }

}

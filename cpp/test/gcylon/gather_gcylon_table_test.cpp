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
        std::vector<std::string> input_file_bases {"../../data/input/gather/numeric_",
                                                   "../../data/input/gather/sales_records_nulls_nunascii_"};
        std::vector<std::vector<std::string>> column_name_vectors{{"0", "1"},
                                                                 {"Country", "Item Type", "Order Date", "Order ID", "Units Sold", "Unit Price"}};
        std::vector<std::vector<std::string>> date_column_vectors{{},
                                                                  {"Order Date"}};

        for(long unsigned int i = 0; i < input_file_bases.size(); i++) {
            std::string input_file_base = input_file_bases[i];
            std::vector<std::string> all_input_files = gcylon::test::constructInputFiles(input_file_base, WORLD_SZ);
            std::string input_filename = all_input_files[RANK];
            std::ifstream in_file(input_filename);
            REQUIRE((in_file.good()));

            for(bool gather_root_table: gather_from_root){
                for(int gather_root: gather_roots) {
                    if (gather_root_table) {
                        REQUIRE((gcylon::test::PerformGatherTest(input_filename,
                                                                 all_input_files,
                                                                 column_name_vectors[i],
                                                                 date_column_vectors[i],
                                                                 gather_root,
                                                                 gather_root_table,
                                                                 ctx)));
                    } else {
                        std::vector<std::string> input_files_no_root = all_input_files;
                        input_files_no_root.erase(input_files_no_root.begin() + gather_root);
                        REQUIRE((gcylon::test::PerformGatherTest(input_filename,
                                                                 input_files_no_root,
                                                                 column_name_vectors[i],
                                                                 date_column_vectors[i],
                                                                 gather_root,
                                                                 gather_root_table,
                                                                 ctx)));
                    }
                }
            }
        }
    }

}

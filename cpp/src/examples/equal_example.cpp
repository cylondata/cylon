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

#include <glog/logging.h>

#include <chrono>
#include <cylon/net/mpi/mpi_communicator.hpp>
#include <cylon/ops.hpp>
#include <cylon/table.hpp>

#include "example_utils.hpp"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        LOG(ERROR) << "./equal_example table_a_len table_b_len" << std::endl;
        return 1;
    }

    auto start_time = std::chrono::steady_clock::now();
    auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
    auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

    std::shared_ptr<cylon::Table> first_table, second_table;

    int a_len = std::stoi(argv[1]);
    int b_len = std::stoi(argv[2]);

    cylon::examples::create_in_memory_tables(a_len, 1, ctx, first_table);
    cylon::examples::create_in_memory_tables(a_len, 1, ctx, second_table);

    bool result = cylon::UnorderedEqual(first_table, second_table);
    bool result2 = cylon::UnorderedEqual(first_table, first_table);

    if (result) {
        std::cout << "equal" << std::endl;
    } else {
        std::cout << "not equal" << std::endl;
    }

    if (result2) {
        std::cout << "equal" << std::endl;
    } else {
        std::cout << "not equal" << std::endl;
    }
    return 0;
}
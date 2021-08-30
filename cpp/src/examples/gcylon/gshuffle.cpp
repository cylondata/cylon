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

#include <iostream>
#include <fstream>
#include <string>

#include <glog/logging.h>
#include <chrono>
#include <thread>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/io/csv.hpp>
#include <cudf/io/types.hpp>
#include <cuda.h>

#include <gcylon/gtable_api.hpp>
#include <gcylon/utils/construct.hpp>
#include <examples/gcylon/print.hpp>
#include <cylon/net/mpi/mpi_communicator.hpp>

using namespace gcylon;
using namespace std::chrono;

int64_t calculateRows(std::string dataSize, const int& cols, int workers) {
    char last_char = dataSize[dataSize.size() - 1];
    char prev_char = dataSize[dataSize.size() - 2];
    int64_t size_num = stoi(dataSize.substr(0, dataSize.size() - 2));
    if (prev_char == 'M' && last_char == 'B') {
        size_num *= 1000000;
    } else if (prev_char == 'G' && last_char == 'B') {
        size_num *= 1000000000;
    } else {
        throw "data size has to end with either MB or GB!";
    }

    return size_num / (cols * workers * 8);
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        LOG(ERROR) << "You must specify the total data size in MB or GB: 100MB, 2GB, etc.";
        return 1;
    }

    const int COLS = 4;
    std::string dataSize = argv[1];
    const bool RESULT_TO_FILE = false;

    auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
    auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
    int my_rank = ctx->GetRank();

    int number_of_GPUs;
    cudaGetDeviceCount(&number_of_GPUs);

    // set the gpu
    cudaSetDevice(my_rank % number_of_GPUs);
    int deviceInUse = -1;
    cudaGetDevice(&deviceInUse);
    LOG(INFO) << "my_rank: "  << my_rank << ", device in use: "<< deviceInUse << ", number of GPUs: " << number_of_GPUs;

    // calculate the number of rows
    int64_t rows = calculateRows(dataSize, COLS, ctx->GetWorldSize());
    LOG(INFO) << "my_rank: "  << my_rank << ", initial dataframe. cols: "<< COLS << ", rows: " << rows;

    std::shared_ptr<cudf::table> tbl = constructTable(COLS, rows);
    auto tv = tbl->view();
    if (my_rank == 0) {
        LOG(INFO) << "my_rank: "  << my_rank << ", initial dataframe................................. ";
        printLongTable(tv);
    }

    // shuffle the table
    std::vector<cudf::size_type> columns_to_hash = {0};
    std::unique_ptr<cudf::table> shuffled_table;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    Shuffle(tv, columns_to_hash, ctx, shuffled_table);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double, std::milli> diff = t2 - t1;
    long int delay = diff.count();

    LOG(INFO) << "my_rank: "  << my_rank << ", duration: "<<  delay;
    auto shuffled_tv = shuffled_table->view();
    if (my_rank == 0) {
        LOG(INFO) << "my_rank: "  << my_rank << ", shuffled dataframe................................. ";
        printLongTable(shuffled_tv);
    }
    LOG(INFO) << "my_rank: "  << my_rank << ", rows in shuffled df: "<< shuffled_tv.num_rows();

    if (RESULT_TO_FILE) {
        std::ofstream srf;
        srf.open("single_run_"s + to_string(my_rank) + ".csv");
        srf << my_rank << "," << delay << endl;
        srf.close();
    }

    ctx->Finalize();
    return 0;
}

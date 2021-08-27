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
    char lastChar = dataSize[dataSize.size() - 1];
    char prevChar = dataSize[dataSize.size() - 2];
    int64_t sizeNum = stoi(dataSize.substr(0, dataSize.size() - 2));
    if (prevChar == 'M' && lastChar == 'B') {
        sizeNum *= 1000000;
    } else if (prevChar == 'G' && lastChar == 'B') {
        sizeNum *= 1000000000;
    } else {
        throw "data size has to end with either MB or GB!";
    }

    return sizeNum / (cols * workers * 8);
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
    int myRank = ctx->GetRank();

    int numberOfGPUs;
    cudaGetDeviceCount(&numberOfGPUs);

    // set the gpu
    cudaSetDevice(myRank % numberOfGPUs);
    int deviceInUse = -1;
    cudaGetDevice(&deviceInUse);
    LOG(INFO) << "myRank: "  << myRank << ", device in use: "<< deviceInUse << ", number of GPUs: " << numberOfGPUs;

    // calculate the number of rows
    int64_t rows = calculateRows(dataSize, COLS, ctx->GetWorldSize());
    LOG(INFO) << "myRank: "  << myRank << ", initial dataframe. cols: "<< COLS << ", rows: " << rows;

    std::shared_ptr<cudf::table> tbl = constructTable(COLS, rows);
    auto tv = tbl->view();
    if (myRank == 0) {
        LOG(INFO) << "myRank: "  << myRank << ", initial dataframe................................. ";
        printLongTable(tv);
    }

    // shuffle the table
    std::vector<cudf::size_type> columns_to_hash = {0};
    std::unique_ptr<cudf::table> shuffledTable;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    Shuffle(tv, columns_to_hash, ctx, shuffledTable);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double, std::milli> diff = t2 - t1;
    long int delay = diff.count();

    LOG(INFO) << "myRank: "  << myRank << ", duration: "<<  delay;
    auto shuffledtv = shuffledTable->view();
    if (myRank == 0) {
        LOG(INFO) << "myRank: "  << myRank << ", shuffled dataframe................................. ";
        printLongTable(shuffledtv);
    }
    LOG(INFO) << "myRank: "  << myRank << ", rows in shuffled df: "<< shuffledtv.num_rows();

    if (RESULT_TO_FILE) {
        std::ofstream srf;
        srf.open("single_run_"s + to_string(myRank) + ".csv");
        srf << myRank << "," << delay << endl;
        srf.close();
    }

    ctx->Finalize();
    return 0;
}

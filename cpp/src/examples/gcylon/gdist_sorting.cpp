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
#include <cudf/sorting.hpp>
#include <cuda.h>

#include <gcylon/gtable_api.hpp>
#include <gcylon/utils/construct.hpp>
#include <examples/gcylon/print.hpp>
#include <cylon/net/mpi/mpi_communicator.hpp>

using namespace gcylon;
using namespace std::chrono;

int64_t calculateRows(std::string dataSize, const int& cols) {
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

    return size_num / (cols * 8);
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        LOG(ERROR) << "You must specify the total data size in MB or GB: 100MB, 2GB, etc.";
        return 1;
    }

    const int COLS = 2;
    std::string dataSize = argv[1];
    const bool RESULT_TO_FILE = false;

    int number_of_GPUs;
    cudaGetDeviceCount(&number_of_GPUs);

    // set the gpu
//    cudaSetDevice();
    int deviceInUse = -1;
    cudaGetDevice(&deviceInUse);
    LOG(INFO) << "device in use: "<< deviceInUse << ", number of GPUs: " << number_of_GPUs;

    // calculate the number of rows
    int64_t rows = calculateRows(dataSize, COLS);

    std::shared_ptr<cudf::table> tbl = constructTable(COLS, rows, 0, true);
//    std::string input_csv_file1 = "data/input/cities_a_0.csv";
//    cudf::io::source_info si1(input_csv_file1);
//    cudf::io::csv_reader_options options1 = cudf::io::csv_reader_options::builder(si1);
//    cudf::io::table_with_metadata tbl_wm = cudf::io::read_csv(options1);
//    auto tbl = tbl_wm.tbl;

    auto tv = tbl->view();
    LOG(INFO) << "initial dataframe................................. ";
    printLongTable(tv);

    // sort the table
    std::vector<cudf::size_type> columns_to_sort = {0};
    std::unique_ptr<cudf::table> sorted_table;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    sorted_table = cudf::sort(tv);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double, std::milli> diff = t2 - t1;
    long int delay = diff.count();
    auto sorted_tv = sorted_table->view();

    LOG(INFO) << "duration: "<<  delay;
    LOG(INFO) << "sorted dataframe................................. ";
    printLongTable(sorted_tv);
    LOG(INFO) << ", rows in sorted df: "<< sorted_tv.num_rows();

    if (RESULT_TO_FILE) {
        LOG(INFO) << "sorted table: number of columns: " << sorted_tv.num_columns() << ", number of rows: " << sorted_tv.num_rows();
        string outFile = string("sorted.csv");
        cudf::io::sink_info sinkInfo(outFile);
        cudf::io::csv_writer_options writerOptions = cudf::io::csv_writer_options::builder(sinkInfo, sorted_tv);
        cudf::io::write_csv(writerOptions);
        LOG(INFO) << "written joined table to the file: " << outFile;
    }

    return 0;
}

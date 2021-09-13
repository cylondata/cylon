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

/**
 * Sorting a cudf table vs merging multiple sorted tables
 * both with the same number of rows
 */

#include <iostream>
#include <fstream>
#include <string>

#include <glog/logging.h>

#include <cudf/table/table.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/sorting.hpp>
#include <cudf/merge.hpp>
#include <cudf/quantiles.hpp>
#include <cuda.h>

#include <gcylon/utils/construct.hpp>
#include <examples/gcylon/print.hpp>

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

int testSorting(const int cols, const int64_t rows) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::unique_ptr<cudf::table> tbl = constructRandomDataTable(cols, rows);
    auto tv = tbl->view();
    LOG(INFO) << "initial dataframe................................. ";
    printLongTable(tv);

    cudaEventRecord(start);
    std::unique_ptr<cudf::table> result_table = cudf::sort(tv);
    cudaEventRecord(stop);
    float fdelay = 0;
    cudaEventElapsedTime(&fdelay, start, stop);
    int delay = (int)fdelay;

    auto result_tv = result_table->view();

    LOG(INFO) << "duration: "<<  delay;
    LOG(INFO) << "sorted dataframe................................. ";
    printLongTable(result_tv);
    LOG(INFO) << ", rows in sorted df: "<< result_tv.num_rows();
    return delay;
}

std::vector<double> quantile_points(int num_of_quantiles) {
    std::vector<double> q;
    double step = 1.0 / num_of_quantiles;
    double qvalue = step;
    q.push_back(qvalue);
    for (int i = 0; i < num_of_quantiles - 2; ++i) {
        qvalue += step;
        q.push_back(qvalue);
    }

    return q;
}

int testQuantiles(const int cols, const int64_t rows) {
    const int num_of_quantiles = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::unique_ptr<cudf::table> tbl = constructRandomDataTable(cols, rows);
    auto tv = tbl->view();
    LOG(INFO) << "initial dataframe................................. ";
    printLongTable(tv);

    std::unique_ptr<cudf::table> result_table;
    std::vector<double> qpoints = quantile_points(num_of_quantiles);
    cudaEventRecord(start);
    result_table = cudf::quantiles(tv, qpoints);
    cudaEventRecord(stop);
    float fdelay = 0;
    cudaEventElapsedTime(&fdelay, start, stop);
    int delay = (int)fdelay;

    auto result_tv = result_table->view();

    LOG(INFO) << "duration: "<<  delay;
    LOG(INFO) << "sorted dataframe................................. ";
    printLongTable(result_tv);
    LOG(INFO) << ", rows in sorted df: "<< result_tv.num_rows();
    return delay;
}

/**
 * merging is broken,
 * single column merging works but produces incorrect result
 * multiple column merging throws an error:
 *      merge: failed to synchronize: cudaErrorIllegalAddress: an illegal memory access was encountered
 * @param cols
 * @param rows
 * @return
 */
int testMerging(const int cols, const int64_t rows) {
    const int num_of_sorted_tables = 2;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<cudf::table_view> tables;
    std::vector<cudf::size_type> key_cols;
    std::vector<cudf::order> column_orders;

    int64_t rows_per_table = rows / num_of_sorted_tables;
    int64_t data_start = 0;
    int step = 1;
    for (int i = 0; i < num_of_sorted_tables; ++i) {
        std::unique_ptr<cudf::table> tbl = constructTable(cols, rows_per_table, data_start, step, true);
        data_start += rows_per_table * cols * step;
        auto tv = tbl->view();
        LOG(INFO) << "initial dataframe................................. ";
        printLongTable(tv);
        tables.push_back(tv);
    }
//    rmm::cuda_stream_default.synchronize();
    cudaDeviceSynchronize();

    for (int i = 0; i < cols; ++i) {
        key_cols.push_back(i);
        column_orders.push_back(cudf::order::ASCENDING);
    }

    cudaEventRecord(start);
    std::unique_ptr<cudf::table> result_table = cudf::merge(tables, key_cols, column_orders);
    cudaEventRecord(stop);
    float fdelay = 0;
    cudaEventElapsedTime(&fdelay, start, stop);
    int delay = (int)fdelay;

    auto result_tv = result_table->view();

    LOG(INFO) << "duration: "<<  delay;
    LOG(INFO) << "sorted dataframe................................. ";
    printLongTable(result_tv);
    LOG(INFO) << ", rows in sorted df: "<< result_tv.num_rows();
    return delay;
}

const bool RESULT_TO_FILE = true;
string OUT_FILE = "single_run.csv";

int main(int argc, char *argv[]) {

    if (argc != 4) {
        LOG(ERROR) << "required three params (sort/merge/quantiles dataSize num_of_columns): sorting 1GB 2, \n"
            << "dataSize in MB or GB: 100MB, 2GB, etc.";
        return 1;
    }

    std::string op_type = argv[1];
    if(op_type != "sort" && op_type != "merge" && op_type != "quantiles") {
        LOG(ERROR) << "first parameter can be either 'sort' or 'merg' or 'quantiles'";
        return 1;
    }

    std::string dataSize = argv[2];
    int cols = stoi(argv[3]);

    int number_of_GPUs;
    cudaGetDeviceCount(&number_of_GPUs);

    // set the gpu
//    cudaSetDevice();
    int deviceInUse = -1;
    cudaGetDevice(&deviceInUse);
    LOG(INFO) << "device in use: "<< deviceInUse << ", number of GPUs: " << number_of_GPUs;

    // calculate the number of rows
    int64_t rows = calculateRows(dataSize, cols);
    LOG(INFO) << "number of columns: "<< cols << ", total number of rows: " << rows;

    int delay = -1;
    if(op_type == "sort")
        delay = testSorting(cols, rows);
    else if(op_type == "merge")
        delay = testMerging(cols, rows);
    else if(op_type == "quantiles")
        delay = testQuantiles(cols, rows);

    if (RESULT_TO_FILE) {
        std::ofstream srf;
        srf.open(OUT_FILE);
        srf << delay << endl;
        srf.close();
    }

    return 0;
}

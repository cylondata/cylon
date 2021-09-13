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

#ifndef GCYLON_EX_CONSTRUCT_H
#define GCYLON_EX_CONSTRUCT_H

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/io/types.hpp>
#include <cuda.h>

#include <stdlib.h>
#include <time.h>

using namespace std;

/**
 * construct a long sorted column with sequentially increasing values
 * @param size
 * @param value_start
 * @return
 */
std::unique_ptr<cudf::column> constructLongColumn(int64_t size, int64_t value_start = 0, int step = 1) {
    int64_t * cpu_buf = new int64_t[size];
    for(int64_t i = 0; i < size; i++) {
        cpu_buf[i] = value_start;
        value_start += step;
    }

    // allocate byte buffer on gpu
    rmm::device_buffer rmm_buf(size * sizeof(int64_t), rmm::cuda_stream_default);

    // copy array to gpu
    auto result = cudaMemcpy(rmm_buf.data(), cpu_buf, size * sizeof(int64_t), cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cout << cudaGetErrorString(result) << endl;
        return nullptr;
    }

    delete [] cpu_buf;
    cudf::data_type dt(cudf::type_id::INT64);
    auto col = std::make_unique<cudf::column>(dt, size, std::move(rmm_buf));
    return col;
}

/**
 * construct a long columns with sequentailly increasing values
 * @param size
 * @param value_start
 * @return
 */
std::unique_ptr<cudf::column> constructRandomLongColumn(int64_t size, int seed) {
    int64_t * cpu_buf = new int64_t[size];

    /* initialize random seed: */
    srand (seed);
    for(int64_t i=0; i < size; i++) {
        cpu_buf[i] = rand();
    }

    // allocate byte buffer on gpu
    rmm::device_buffer rmm_buf(size * sizeof(int64_t), rmm::cuda_stream_default);
    // copy array to gpu
    auto result = cudaMemcpy(rmm_buf.data(), cpu_buf, size * sizeof(int64_t), cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cout << cudaGetErrorString(result) << endl;
        return nullptr;
    }

    delete [] cpu_buf;
    cudf::data_type dt(cudf::type_id::INT64);
    auto col = std::make_unique<cudf::column>(dt, size, std::move(rmm_buf));
    return col;
}

/**
 * construct a sorted cudf Table with int64_t data
 * data is sequentially increasing
 * @param columns number of columns in the table
 * @param rows number of rows in the table
 * @param value_start start value of the first columns
 * @param step value increase amount
 * @param cont continue to increase values sequentially when moving from one column to another
 * @return
 */
std::unique_ptr<cudf::table> constructTable(int columns,
                                            int64_t rows,
                                            int64_t value_start = 0,
                                            int step = 1,
                                            bool cont=false) {

    std::vector<std::unique_ptr<cudf::column>> column_vector{};
    for (int i=0; i < columns; i++) {
        std::unique_ptr<cudf::column> col = constructLongColumn(rows, value_start, step);
        column_vector.push_back(std::move(col));
        if(cont)
            value_start += rows * step;
    }

    return std::make_unique<cudf::table>(std::move(column_vector));
}

/**
 * construct a cudf Table with int64_t data
 * @param columns number of columns in the table
 * @param rows number of rows in the table
 * @param value_start start value of the first columns
 * @param cont continue to increase values sequentially when moving from one column to another
 * @return
 */
std::unique_ptr<cudf::table> constructRandomDataTable(int columns, int64_t rows) {

    std::vector<std::unique_ptr<cudf::column>> column_vector{};
    for (int i=0; i < columns; i++) {
        std::unique_ptr<cudf::column> col = constructRandomLongColumn(rows, i*10);
        column_vector.push_back(std::move(col));
    }

    return std::make_unique<cudf::table>(std::move(column_vector));
}

#endif //GCYLON_EX_CONSTRUCT_H

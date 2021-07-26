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

using namespace std;

std::unique_ptr<cudf::column> constructLongColumn(int64_t size, int64_t valueStart = 0) {
    int64_t * cpuBuf = new int64_t[size];
    for(int64_t i=0; i < size; i++)
        cpuBuf[i] = valueStart++;

    // allocate byte buffer on gpu
    rmm::device_buffer rmmBuf(size * 8, rmm::cuda_stream_default);
    // copy array to gpu
    auto result = cudaMemcpy(rmmBuf.data(), cpuBuf, size * 8, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cout << cudaGetErrorString(result) << endl;
        return nullptr;
    }

    delete [] cpuBuf;
    cudf::data_type dt(cudf::type_id::INT64);
    auto col = std::make_unique<cudf::column>(dt, size, std::move(rmmBuf));
    return col;
}

std::shared_ptr<cudf::table> constructTable(int columns, int64_t rows, int64_t valueStart = 0, bool cont=false) {

    std::vector<std::unique_ptr<cudf::column>> columnVector{};
    for (int i=0; i < columns; i++) {
        std::unique_ptr<cudf::column> col = constructLongColumn(rows, valueStart);
        columnVector.push_back(std::move(col));
        if(cont)
            valueStart += rows;
    }

    return std::make_shared<cudf::table>(std::move(columnVector));
}
#endif //GCYLON_EX_CONSTRUCT_H

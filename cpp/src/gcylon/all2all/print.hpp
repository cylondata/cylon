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

#ifndef CYLON_ALL2ALL_UTIL_H
#define CYLON_ALL2ALL_UTIL_H

#include <bitset>
#include <cudf/types.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/null_mask.hpp>

using std::cout;
using std::endl;
using std::string;

inline void printColumn(cudf::column_view const& input, int columnIndex, int start, int end) {
    cout << "column[" << columnIndex << "]:  ";
    if (input.type().id() == cudf::type_id::STRING) {
        cudf::strings_column_view scv(input);
        cout << "Can't print string column :(((( " << endl;
//        cudf::strings::print(scv);
//        printStringColumnA(input, columnIndex);
        return;
    }

    if (input.type().id() == cudf::type_id::INT32 ||
            input.type().id() == cudf::type_id::INT64 ||
            input.type().id() == cudf::type_id::FLOAT32 ||
            input.type().id() == cudf::type_id::FLOAT64) {

        const uint32_t * nullMask = input.null_mask();
        for (cudf::size_type i = start; i < end; ++i) {
            if (input.nullable() && cudf::count_set_bits(nullMask, i, i+1) == 0) {
                cout << "NULL";
            } else {
                switch (input.type().id() ) {
                    case cudf::type_id::INT32:
                        cout << cudf::detail::get_value<int32_t>(input, i, rmm::cuda_stream_default); break;
                    case cudf::type_id::INT64:
                        cout << cudf::detail::get_value<int64_t>(input, i, rmm::cuda_stream_default); break;
                    case cudf::type_id::FLOAT64:
                        cout << cudf::detail::get_value<double>(input, i, rmm::cuda_stream_default); break;
                    case cudf::type_id::FLOAT32:
                        cout << cudf::detail::get_value<float>(input, i, rmm::cuda_stream_default); break;
                }
            }
            cout << ", ";
        }
        cout << endl;
        return;
    }

    // if it arrives to this point
    cout << "data type is not supported. data-type id: " << cudf::type_dispatcher(input.type(), cudf::type_to_name{}) << endl;
}

inline void printColumn(cudf::column_view const& input, int columnIndex) {
    printColumn(input, columnIndex, 0, input.size());
}

inline void printIntColumnPart(const uint8_t * buff, int columnIndex, int start, int end) {
    std::cout << "column[" << columnIndex << "][" << start << "-" << end << "]: ";
    int8_t *hostArray= new int8_t [(end - start)];
    cudaMemcpy(hostArray, buff, (end-start), cudaMemcpyDeviceToHost);
    int32_t * hdata = (int32_t *) hostArray;
    int size = (end-start)/4;

    for (int i = start; i < size; ++i) {
        std::cout << hdata[i] << ", ";
    }
    std::cout << std::endl;
}

inline void printLongColumnPart(const uint8_t * buff, int columnIndex, int start, int end) {
    cout << "column[" << columnIndex << "][" << start << "-" << end << "]: ";
    uint8_t *hostArray= new uint8_t[end-start];
    cudaMemcpy(hostArray, buff, end-start, cudaMemcpyDeviceToHost);
    int64_t * hdata = (int64_t *) hostArray;
    int size = (end-start)/8;

    for (int i = start; i < size; ++i) {
        cout << hdata[i] << ", ";
    }
    cout << endl;
}

inline void printStringColumnPart(const uint8_t * buff, int columnIndex, int start, int end) {
    std::cout << "column[" << columnIndex << "][" << start << "-" << end << "]: ";
    char *hostArray= new char[end - start + 1];
    cudaMemcpy(hostArray, buff, end-start, cudaMemcpyDeviceToHost);
    hostArray[end-start] = '\0';
    std::cout << hostArray << std::endl;
}

inline void printStringColumnPart(cudf::column_view const& cv, int columnIndex, int start, int end) {
    cudf::strings_column_view scv(cv);
    int startIndex = cudf::detail::get_value<int32_t>(scv.offsets(), start, rmm::cuda_stream_default);
    int endIndex = cudf::detail::get_value<int32_t>(scv.offsets(), end, rmm::cuda_stream_default);
    printStringColumnPart(scv.chars().data<uint8_t>() + startIndex, columnIndex,  startIndex, endIndex);
}

inline void printStringColumn(cudf::column_view const& cv, int columnIndex) {
    cudf::strings_column_view scv(cv);
    int endIndex = cudf::detail::get_value<int32_t>(scv.offsets(), scv.offsets().size() - 1, rmm::cuda_stream_default);
    printStringColumnPart(scv.chars().data<uint8_t>(), columnIndex,  0, endIndex);
//    printIntColumnPartA(scv.offsets().data<uint8_t>(), columnIndex, 0, scv.offsets().size()*4);
}

inline void printCharArray(cudf::column_view cv, int charSize) {
    char *hostArray= new char[charSize+1];
    cudaMemcpy(hostArray, cv.data<char>(), charSize, cudaMemcpyDeviceToHost);
    hostArray[charSize] = '\0';
    std::cout << "chars:: " << hostArray << std::endl;
}

inline void printNullMask(const cudf::column_view &cv) {
    if (!cv.nullable()) {
        std::cout << "the column is not nullable ......................: " << std::endl;
        return;
    }
    std::cout << "number of nulls in the column: " << cv.null_count() << std::endl;
    std::size_t size = cudf::bitmask_allocation_size_bytes(cv.size());
    uint8_t *hostArray= new uint8_t[size];
    cudaMemcpy(hostArray, (uint8_t *)cv.null_mask(), size, cudaMemcpyDeviceToHost);
    for (std::size_t i = 0; i < size; ++i) {
        std::bitset<8> x(hostArray[i]);
        std::cout << i << ":" << x << " ";
    }
    std::cout << std::endl;
}

inline void printTableColumnTypes(cudf::table_view & tview) {
    cout << "number of columns: " << tview.num_columns() << ", number of rows: " << tview.num_rows() << endl;

    for (int i = 0; i < tview.num_columns(); ++i) {
        cudf::column_view cv = tview.column(i);
        cout << "column: " << i << ", data-type:" << cudf::type_dispatcher(cv.type(), cudf::type_to_name{});
        if (cv.nullable()) {
            cout << ", nullable" << endl;
        }else
            cout << ", not nullable" << endl;
    }

}

#endif //CYLON_ALL2ALL_UTIL_H

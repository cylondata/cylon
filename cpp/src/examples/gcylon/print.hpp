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

#ifndef GCYLON_EX_PRINT_H
#define GCYLON_EX_PRINT_H

#include <iostream>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/io/types.hpp>
#include <cuda.h>

#include <gcylon/utils/util.hpp>

using namespace gcylon;

void printLongColumn(const cudf::column_view &cv, int64_t topN = 5, int64_t tailN = 5) {
    if(cv.size() < (topN + tailN)) {
        std::cout << "!!!!!!!!!!!!!!!! number of elements in the column is less than (topN + tailN)";
        return;
    }

    int64_t * hdata = getColumnTop<int64_t>(cv, topN);
    std::cout << "Top: " << topN << " elements of the column: " << std::endl;
    for (int i = 0; i < topN; ++i) {
        std::cout << i << ": " << hdata[i] << std::endl;
    }

    hdata = getColumnTail<int64_t>(cv, tailN);
    std::cout << "Tail: " << tailN << " elements of the column: " << std::endl;
    int64_t ci = cv.size() - tailN;
    for (int i = 0; i < tailN; ++i) {
        std::cout << ci++ << ": " << hdata[i] << std::endl;
    }
}

void printWholeTable(cudf::table_view &tableView) {
    // get column tops
    std::vector<int64_t *> columnTops{};
    for (int i = 0; i < tableView.num_columns(); ++i) {
        columnTops.push_back(getColumnTop<int64_t>(tableView.column(i), tableView.num_rows()));
    }

    std::cout << "..................................................................................." << std::endl;
    // print header
    for (int i = 0; i < tableView.num_columns(); ++i) {
        std::cout << "\t\t" << i;
    }
    std::cout << std::endl;

    for (int i=0; i<tableView.num_rows(); i++) {
        std::cout << i;
        for (auto columnTop: columnTops) {
            std::cout << "\t\t" << columnTop[i];
        }
        std::cout << std::endl;
    }
    std::cout << "..................................................................................." << std::endl;
}

void printLongTable(cudf::table_view &tableView, int64_t topN = 5, int64_t tailN = 5) {
    // get column tops
    std::vector<int64_t *> columnTops{};
    for (int i = 0; i < tableView.num_columns(); ++i) {
        columnTops.push_back(getColumnTop<int64_t>(tableView.column(i), topN));
    }

    std::cout << "..................................................................................." << std::endl;
    // print table top
    // print header
    for (int i = 0; i < tableView.num_columns(); ++i) {
        std::cout << "\t\t" << i;
    }
    std::cout << std::endl;

    for (int i=0; i<topN; i++) {
        std::cout << i;
        for (auto columnTop: columnTops) {
            std::cout << "\t\t" << columnTop[i];
        }
        std::cout << std::endl;
    }
    // print table tail
    std::cout << "......................................" << std::endl;
    std::vector<int64_t *> columnTails{};
    for (int i = 0; i < tableView.num_columns(); ++i) {
        columnTails.push_back(getColumnTail<int64_t>(tableView.column(i), tailN));
    }

    int64_t ci = tableView.num_rows() - tailN;
    for (int i=0; i<tailN; i++) {
        std::cout << ci++;
        for (auto columnTail: columnTails) {
            std::cout << "\t\t" << columnTail[i];
        }
        std::cout << std::endl;
    }
    std::cout << "..................................................................................." << std::endl;
}
#endif //GCYLON_EX_PRINT_H

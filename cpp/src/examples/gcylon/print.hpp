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
#include <string>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/io/types.hpp>
#include <cuda.h>

#include <gcylon/utils/util.hpp>

using namespace std;

void printLongColumn(const cudf::column_view &cv, int64_t topN = 5, int64_t tailN = 5) {
    if(cv.size() < (topN + tailN)) {
        cout << "!!!!!!!!!!!!!!!! number of elements in the column is less than (topN + tailN)";
        return;
    }

    int64_t * hdata = getColumnTop<int64_t>(cv, topN);
    cout << "Top: " << topN << " elements of the column: " << endl;
    for (int i = 0; i < topN; ++i) {
        cout << i << ": " << hdata[i] << endl;
    }

    hdata = getColumnTail<int64_t>(cv, tailN);
    cout << "Tail: " << tailN << " elements of the column: " << endl;
    int64_t ci = cv.size() - tailN;
    for (int i = 0; i < tailN; ++i) {
        cout << ci++ << ": " << hdata[i] << endl;
    }
}

void printWholeTable(cudf::table_view &tableView) {
    // get column tops
    std::vector<int64_t *> columnTops{};
    for (int i = 0; i < tableView.num_columns(); ++i) {
        columnTops.push_back(getColumnTop<int64_t>(tableView.column(i), tableView.num_rows()));
    }

    cout << "..................................................................................." << endl;
    // print header
    for (int i = 0; i < tableView.num_columns(); ++i) {
        cout << "\t\t" << i;
    }
    cout << endl;

    for (int i=0; i<tableView.num_rows(); i++) {
        cout << i;
        for (auto columnTop: columnTops) {
            cout << "\t\t" << columnTop[i];
        }
        cout << endl;
    }
    cout << "..................................................................................." << endl;
}

void printLongTable(cudf::table_view &tableView, int64_t topN = 5, int64_t tailN = 5) {
    // get column tops
    std::vector<int64_t *> columnTops{};
    for (int i = 0; i < tableView.num_columns(); ++i) {
        columnTops.push_back(getColumnTop<int64_t>(tableView.column(i), topN));
    }

    cout << "..................................................................................." << endl;
    // print table top
    // print header
    for (int i = 0; i < tableView.num_columns(); ++i) {
        cout << "\t\t" << i;
    }
    cout << endl;

    for (int i=0; i<topN; i++) {
        cout << i;
        for (auto columnTop: columnTops) {
            cout << "\t\t" << columnTop[i];
        }
        cout << endl;
    }
    // print table tail
    cout << "......................................" << endl;
    std::vector<int64_t *> columnTails{};
    for (int i = 0; i < tableView.num_columns(); ++i) {
        columnTails.push_back(getColumnTail<int64_t>(tableView.column(i), tailN));
    }

    int64_t ci = tableView.num_rows() - tailN;
    for (int i=0; i<tailN; i++) {
        cout << ci++;
        for (auto columnTail: columnTails) {
            cout << "\t\t" << columnTail[i];
        }
        cout << endl;
    }
    cout << "..................................................................................." << endl;
}
#endif //GCYLON_EX_PRINT_H

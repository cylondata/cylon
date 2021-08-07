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

#ifndef GCYLON_TEST_UTILS_HPP_
#define GCYLON_TEST_UTILS_HPP_

#include <glog/logging.h>
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/types.hpp>

#include <gcylon/gtable.hpp>
#include <gcylon/utils/util.hpp>

// this is a toggle to generate test files. Set execute to 0 then, it will generate the expected
// output files
#define EXECUTE 1

using namespace gcylon;

namespace gcylon {
namespace test {

cudf::io::table_with_metadata readCSV(std::string & filename, int rank) {
    cudf::io::source_info si(filename);
    cudf::io::csv_reader_options options = cudf::io::csv_reader_options::builder(si);
    cudf::io::table_with_metadata ctable = cudf::io::read_csv(options);
    LOG(INFO) << "myRank: "  << rank << ", input file: " << filename
        << ", cols: "<< ctable.tbl->view().num_columns() << ", rows: " << ctable.tbl->view().num_rows();
    return ctable;
}

void writeCSV(cudf::table_view & tv, std::string filename, int rank, cudf::io::table_metadata &tableMetadata) {
    cudf::io::sink_info sinkInfo(filename);
    cudf::io::csv_writer_options writerOptions =
            cudf::io::csv_writer_options::builder(sinkInfo, tv).metadata(&tableMetadata);
    LOG(INFO) << "myRank: "  << rank << ", output file: " << filename
        << ", cols: "<< tv.num_columns() << ", rows: " << tv.num_rows();
    cudf::io::write_csv(writerOptions);
}

bool PerformShuffleTest(string & inputFileName, string & outputFileName, int shuffleIndex, int rank) {
    cudf::io::table_with_metadata inputTable = readCSV(inputFileName, rank);
    auto inputTv = inputTable.tbl->view();

    // shuffle the table
    std::vector<cudf::size_type> columns_to_hash = {shuffleIndex};
    std::unique_ptr<cudf::table> shuffledTable;
    Shuffle(inputTv, columns_to_hash, ctx, shuffledTable);
    auto shuffledtv = shuffledTable->view();

#if EXECUTE
    cudf::io::table_with_metadata savedShuffledTable = readCSV(outputFileName, rank);
    auto savedTv = savedShuffledTable.tbl->view();
    return equal(shuffledtv, savedTv);
#else
    writeCSV(shuffledtv, outputFileName, rank, inputTable.metadata);
    return true;
#endif
}



}
}
#endif //CYLON_CPP_SRC_EXAMPLES_TEST_UTILS_HPP_

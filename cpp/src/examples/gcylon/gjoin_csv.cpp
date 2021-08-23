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

#include <glog/logging.h>
#include <cudf/table/table.hpp>
#include <cudf/io/csv.hpp>

#include <gcylon/gtable.hpp>
#include <gcylon/gtable_api.hpp>
#include <cylon/net/mpi/mpi_communicator.hpp>

using std::cout;
using std::endl;
using std::string;
using namespace gcylon;

int main(int argc, char *argv[]) {

    if (argc != 3) {
        std::cout << "You must specify two CSV input files.\n";
        return 1;
    }

    std::string input_csv_file1 = argv[1];
    std::string input_csv_file2 = argv[2];

    auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
    auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
    int myRank = ctx->GetRank();

    LOG(INFO) << "myRank: "  << myRank << ", world size: " << ctx->GetWorldSize();

    int numberOfGPUs;
    cudaGetDeviceCount(&numberOfGPUs);
    LOG(INFO) << "myRank: "  << myRank << ", number of GPUs: " << numberOfGPUs;

    // set the gpu
    cudaSetDevice(myRank % numberOfGPUs);

    // construct table1
    cudf::io::source_info si1(input_csv_file1);
    cudf::io::csv_reader_options options1 = cudf::io::csv_reader_options::builder(si1);
    cudf::io::table_with_metadata ctable1 = cudf::io::read_csv(options1);
    LOG(INFO) << myRank << ", " << input_csv_file1 << ", number of columns: "
              << ctable1.tbl->num_columns() << ", number of rows: " << ctable1.tbl->num_rows();

    std::shared_ptr<GTable> sourceGTable1;
    cylon::Status status = GTable::FromCudfTable(ctx, ctable1.tbl, sourceGTable1);
    if (!status.is_ok()) {
        LOG(ERROR) << "GTable is not constructed successfully.";
        ctx->Finalize();
        return 1;
    }

    // construct table2
    cudf::io::source_info si2(input_csv_file2);
    cudf::io::csv_reader_options options2 = cudf::io::csv_reader_options::builder(si2);
    cudf::io::table_with_metadata ctable2 = cudf::io::read_csv(options2);
    LOG(INFO) << myRank << ", " << input_csv_file2 << ", number of columns: "
              << ctable2.tbl->num_columns() << ", number of rows: " << ctable2.tbl->num_rows();

    std::shared_ptr<GTable> sourceGTable2;
    status = GTable::FromCudfTable(ctx, ctable2.tbl, sourceGTable2);
    if (!status.is_ok()) {
        LOG(ERROR) << "GTable is not constructed successfully.";
        ctx->Finalize();
        return 1;
    }

    // join the tables on the first columns
    std::shared_ptr<GTable> joinedGTable;
    auto join_config = cylon::join::config::JoinConfig(cylon::join::config::JoinType::FULL_OUTER,
                                                       0,
                                                       0,
                                                       cylon::join::config::JoinAlgorithm::HASH);
    status = DistributedJoin(sourceGTable1, sourceGTable2, join_config, joinedGTable);
    if (!status.is_ok()) {
        LOG(ERROR) << "GTable is not joined successfully.";
        ctx->Finalize();
        return 1;
    }
    cudf::table_view tv = joinedGTable->GetCudfTable()->view();

    // write the results to a file
    if (tv.num_rows() == 0) {
        LOG(INFO) << myRank << ": joined table is empty";
    } else {
        LOG(INFO) << myRank << ", Joined table: number of columns: " << tv.num_columns() << ", number of rows: " << tv.num_rows();
        string outFile = string("joined") + std::to_string(myRank) + ".csv";
        cudf::io::sink_info sinkInfo(outFile);
        cudf::io::csv_writer_options writerOptions = cudf::io::csv_writer_options::builder(sinkInfo, tv);
        cudf::io::write_csv(writerOptions);
        cout << myRank << ", written joined table to the file: " << outFile << endl;
    }

    ctx->Finalize();
    return 0;
}

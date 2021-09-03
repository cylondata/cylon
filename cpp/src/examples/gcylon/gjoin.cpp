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
#include <cudf/table/table_view.hpp>

#include <gcylon/utils/construct.hpp>
#include <gcylon/gtable_api.hpp>
#include <cylon/net/mpi/mpi_communicator.hpp>
#include <examples/gcylon/print.hpp>

using std::string;
using namespace gcylon;

int main(int argc, char *argv[]) {

    const int COLS = 2;
    const int64_t ROWS = 10;

    auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
    auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
    int my_rank = ctx->GetRank();

    LOG(INFO) << "my_rank: "  << my_rank << ", world size: " << ctx->GetWorldSize();

    int number_os_GPUs;
    cudaGetDeviceCount(&number_os_GPUs);
    LOG(INFO) << "my_rank: "  << my_rank << ", number of GPUs: " << number_os_GPUs << endl;

    // set the gpu
    cudaSetDevice(my_rank % number_os_GPUs);

    // construct table1
    int64_t start = my_rank * 100;
    std::shared_ptr<cudf::table> tbl1 = constructTable(COLS, ROWS, start, true);
    auto tv1 = tbl1->view();
    LOG(INFO) << "my_rank: "  << my_rank << ", initial dataframe. cols: "<< tv1.num_columns() << ", rows: " << tv1.num_rows();
    printWholeTable(tv1);

    // construct table1
    std::shared_ptr<cudf::table> tbl2 = constructTable(COLS, ROWS, start + 5, true);
    auto tv2 = tbl2->view();
    LOG(INFO) << "my_rank: "  << my_rank << ", initial dataframe. cols: "<< tv2.num_columns() << ", rows: " << tv2.num_rows();
    printWholeTable(tv2);

    // join the tables
    std::unique_ptr<cudf::table> joined_table;
    auto join_config = cylon::join::config::JoinConfig(cylon::join::config::JoinType::INNER,
                                                       0,
                                                       0,
                                                       cylon::join::config::JoinAlgorithm::HASH);
    cylon::Status status = DistributedJoin(tv1, tv2, join_config, ctx, joined_table);
    if (!status.is_ok()) {
        LOG(INFO) << "Joining tables failed.";
        ctx->Finalize();
        return 1;
    }
    auto tvj = joined_table->view();

    if (tvj.num_rows() == 0) {
        LOG(INFO) << my_rank << ": joined table is empty";
    } else {
        LOG(INFO) << my_rank << ", joined table. number of columns: " << tvj.num_columns() << ", number of rows: " << tvj.num_rows();
        printWholeTable(tvj);
    }

    ctx->Finalize();
    return 0;
}

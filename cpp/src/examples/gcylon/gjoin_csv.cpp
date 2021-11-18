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

using std::string;
using namespace gcylon;

int main(int argc, char *argv[]) {

  if (argc != 3) {
    LOG(ERROR) << "You must specify two CSV input files.";
    return 1;
  }

  std::string input_csv_file1 = argv[1];
  std::string input_csv_file2 = argv[2];

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
  int my_rank = ctx->GetRank();

  LOG(INFO) << "my_rank: " << my_rank << ", world size: " << ctx->GetWorldSize();

  int number_of_GPUs;
  cudaGetDeviceCount(&number_of_GPUs);
  LOG(INFO) << "my_rank: " << my_rank << ", number of GPUs: " << number_of_GPUs;

  // set the gpu
  cudaSetDevice(my_rank % number_of_GPUs);

  // construct table1
  cudf::io::source_info si1(input_csv_file1);
  cudf::io::csv_reader_options options1 = cudf::io::csv_reader_options::builder(si1);
  cudf::io::table_with_metadata ctable1 = cudf::io::read_csv(options1);
  LOG(INFO) << my_rank << ", " << input_csv_file1 << ", number of columns: "
            << ctable1.tbl->num_columns() << ", number of rows: " << ctable1.tbl->num_rows();

  std::shared_ptr<GTable> source_table1;
  cylon::Status status = GTable::FromCudfTable(ctx, ctable1.tbl, source_table1);
  if (!status.is_ok()) {
    LOG(ERROR) << "GTable is not constructed successfully.";
    ctx->Finalize();
    return 1;
  }

  // construct table2
  cudf::io::source_info si2(input_csv_file2);
  cudf::io::csv_reader_options options2 = cudf::io::csv_reader_options::builder(si2);
  cudf::io::table_with_metadata ctable2 = cudf::io::read_csv(options2);
  LOG(INFO) << my_rank << ", " << input_csv_file2 << ", number of columns: "
            << ctable2.tbl->num_columns() << ", number of rows: " << ctable2.tbl->num_rows();

  std::shared_ptr<GTable> source_table2;
  status = GTable::FromCudfTable(ctx, ctable2.tbl, source_table2);
  if (!status.is_ok()) {
    LOG(ERROR) << "GTable is not constructed successfully.";
    ctx->Finalize();
    return 1;
  }

  // join the tables on the first columns
  std::shared_ptr<GTable> joined_table;
  auto join_config = cylon::join::config::JoinConfig(cylon::join::config::JoinType::FULL_OUTER,
                                                     0,
                                                     0,
                                                     cylon::join::config::JoinAlgorithm::HASH);
  status = DistributedJoin(source_table1, source_table2, join_config, joined_table);
  if (!status.is_ok()) {
    LOG(ERROR) << "GTable is not joined successfully.";
    ctx->Finalize();
    return 1;
  }
  cudf::table_view tv = joined_table->GetCudfTable()->view();

  // write the results to a file
  if (tv.num_rows() == 0) {
    LOG(INFO) << my_rank << ": joined table is empty";
  } else {
    LOG(INFO) << my_rank << ", Joined table: number of columns: " << tv.num_columns() << ", number of rows: "
              << tv.num_rows();
    string outFile = string("joined") + std::to_string(my_rank) + ".csv";
    cudf::io::sink_info sinkInfo(outFile);
    cudf::io::csv_writer_options writerOptions = cudf::io::csv_writer_options::builder(sinkInfo, tv);
    cudf::io::write_csv(writerOptions);
    LOG(INFO) << my_rank << ", written joined table to the file: " << outFile;
  }

  ctx->Finalize();
  return 0;
}

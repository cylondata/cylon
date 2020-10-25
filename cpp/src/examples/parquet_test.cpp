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

#include <table.hpp>
#include <net/mpi/mpi_communicator.hpp>

int main(int argc, char *argv[]) {
    auto start_start = std::chrono::steady_clock::now();
    auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
    auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

    std::string cmakePath = EXAMPLE_CMAKE_DIR;
    std::string pathFromSrc = "/../../../data/input/parquet1_0.parquet";
    std::string pathToOutput = "/../../../data/input/parquet1_0_1.parquet";
    std::string fullSrcPath = cmakePath + pathFromSrc;
    std::string fullOutputPath = cmakePath + pathToOutput;

    std::shared_ptr<cylon::Table> first_table;
    auto status = cylon::FromParquet(ctx, fullSrcPath, first_table);
    if (!status.is_ok()) {
        LOG(INFO) << "Table reading failed " << status.get_msg();
        ctx->Finalize();
        return 1;
    }

    auto read_end_time = std::chrono::steady_clock::now();

    LOG(INFO) << "Read tables in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                      read_end_time - start_start).count() << "[ms]";

    LOG(INFO) << "Table Data";

    first_table->Print();

    auto parquetOptions = cylon::io::config::ParquetOptions().ChunkSize(5);
    first_table->WriteParquet(ctx, fullOutputPath, parquetOptions);
    return 0;
}
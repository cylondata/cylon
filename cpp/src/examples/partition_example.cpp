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
#include <net/mpi/mpi_communicator.hpp>
#include <ctx/cylon_context.hpp>
#include <util/builtins.hpp>
#include <table.hpp>
#include <chrono>
#include <arrow/api.h>
#include <arrow/array.h>
#include <random>
#include <mpi.h>

#include <arrow/compute/api.h>
#include <ctx/arrow_memory_pool_utils.hpp>

#include <util/arrow_utils.hpp>
#include <groupby/groupby.hpp>

#include <arrow/arrow_partition_kernels.hpp>
#include <partition/partition.hpp>

void create_table(char *const *argv,
                  arrow::MemoryPool *pool,
                  std::shared_ptr<arrow::Table> &left_table);

template<typename T>
void print_vec(std::vector<T> vec) {
  for (auto a: vec) {
    std::cout << a << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    LOG(ERROR) << "There should be 2 args. count, duplication factor";
    return 1;
  }

  auto start_start = std::chrono::steady_clock::now();
  auto mpi_config = cylon::net::MPIConfig::Make();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  arrow::MemoryPool *pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::Table> arrow_table;
//  create_binary_table(argv, ctx, pool, left_table, right_table);
  create_table(argv, pool, arrow_table);

  std::shared_ptr<cylon::Table> table;
  auto status = cylon::Table::FromArrowTable(ctx, arrow_table, table);
  if (!status.is_ok()) {
    LOG(INFO) << "Table reading failed " << argv[1];
    ctx->Finalize();
    return 1;
  }

  auto read_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Read tables in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(read_end_time - start_start).count() << "[ms]";

  std::shared_ptr<cylon::Table> output;

  auto s = cylon::DistributedSort(table, 1, output);
  if (!s.is_ok()) {
    std::cout << "dist sort failed " << s.get_msg() << std::endl;
    return 1;
  }
  std::cout << "sorted table " << ctx->GetRank() << " " << output->Rows() << std::endl;
  cylon::WriteCSV(table, "/tmp/source" + std::to_string(ctx->GetRank()) + ".txt");
  cylon::WriteCSV(output, "/tmp/output" + std::to_string(ctx->GetRank()) + ".txt");

  ctx->Finalize();
  return 0;
}

void create_table(char *const *argv,
                  arrow::MemoryPool *pool,
                  std::shared_ptr<arrow::Table> &left_table) {
  arrow::Int64Builder longb(pool);
  arrow::Int32Builder intb(pool);
  arrow::DoubleBuilder doubleb(pool);
  arrow::FloatBuilder floatb(pool);
  uint64_t count = std::stoull(argv[1]);
  double dup = std::stod(argv[2]);

  std::cout << "#### lines " << count << " dup " << dup << std::endl;

  std::random_device rd;
  std::mt19937_64 gen64(rd());
  std::mt19937 gen32(rd());
  std::uniform_int_distribution<int64_t> longd(0, (int64_t) (count * dup));
//  std::uniform_int_distribution<int64_t> longd(0, (int64_t) (count * dup));
  std::uniform_int_distribution<int32_t> intd(0, (int32_t) (count * dup));

//  std::uniform_real_distribution<double> doubled(0, count * dup);
  std::chi_squared_distribution<double> doubled;
  std::uniform_real_distribution<float> floatd;

  arrow::Status st = longb.Reserve(count);
  st = doubleb.Reserve(count);
  st = intb.Reserve(count);
  st = floatb.Reserve(count);

  for (uint64_t i = 0; i < count; i++) {
    int64_t vl = longd(gen64);
    int32_t vi = intd(gen32);
    double vd = doubled(gen64);
    float vf = floatd(gen32);

    longb.UnsafeAppend(vl);
    intb.UnsafeAppend(vi);
    doubleb.UnsafeAppend(vd);
    floatb.UnsafeAppend(vf);
  }

  std::shared_ptr<arrow::Array> arr0, arr1, arr2, arr3;

  st = longb.Finish(&arr0);
  st = doubleb.Finish(&arr1);
  st = intb.Finish(&arr2);
  st = floatb.Finish(&arr3);

  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
      arrow::field("first", arrow::int64()),
      arrow::field("second", arrow::float64()),
      arrow::field("third", arrow::int32()),
      arrow::field("fourth", arrow::float32())};
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  arrow::ArrayVector av1 = {arr0, arr0};
  arrow::ArrayVector av2 = {arr1, arr1};
  arrow::ArrayVector av3 = {arr2, arr2};
  arrow::ArrayVector av4 = {arr3, arr3};
  const std::shared_ptr<arrow::ChunkedArray> &ca1 = std::make_shared<arrow::ChunkedArray>(av1);
  const std::shared_ptr<arrow::ChunkedArray> &ca2 = std::make_shared<arrow::ChunkedArray>(av2);
  const std::shared_ptr<arrow::ChunkedArray> &ca3 = std::make_shared<arrow::ChunkedArray>(av3);
  const std::shared_ptr<arrow::ChunkedArray> &ca4 = std::make_shared<arrow::ChunkedArray>(av4);
  left_table = arrow::Table::Make(schema, {ca1, ca2, ca3, ca4});

}

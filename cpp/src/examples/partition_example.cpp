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
  MPI_Barrier(MPI_COMM_WORLD);

  std::shared_ptr<cylon::Table> table;
  auto status = cylon::Table::FromArrowTable(ctx, arrow_table, table);
  if (!status.is_ok()) {
    LOG(INFO) << "Table reading failed " << argv[1];
    ctx->Finalize();
    return 1;
  }

  auto read_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Read tables in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                read_end_time - start_start).count() << "[ms]";

//  first_table->WriteCSV("/tmp/source" + std::to_string(ctx->GetRank()) + ".txt");

//  std::shared_ptr<cylon::Table> output;

  int num_partitions = 256;
  int idx_col = 1;
  std::vector<uint32_t> target_partitions(arrow_table->num_rows(), 0);
//  target_partitions.reserve(arrow_table->num_rows());
  std::vector<uint32_t> counts(num_partitions, 0);
//  std::vector<uint32_t> counts(num_partitions, 0);
//  for (const auto &arr: arrow_table->column(0)->chunks()) {
//    const std::shared_ptr<arrow::Int64Array> &carr = std::static_pointer_cast<arrow::Int64Array>(arr);
//    for(int64_t i = 0; i < carr->length(); i++){
//      int32_t p = carr->Value(i) % num_partitions;
//      target_partitions.push_back(p);
//      counts[p]++;
//    }
//  }
//  cylon::ModuloPartitionKernel<arrow::Int64Type> kern(num_partitions);
//  for (int i = 0; i < 8; i++) {
//    target_partitions.clear();
//    counts.clear();
//    auto s = kern.Partition(arrow_table->column(0), target_partitions, counts);
//    if (!s.is_ok()) return 1;
//  }

//  for (int i = 0; i < 4; i++) {
//    std::fill(target_partitions.begin(), target_partitions.end(), 0);
//    std::fill(counts.begin(), counts.end(), 0);
//
//    auto s = cylon::ApplyPartition(table, {3, 2, 1, 0}, num_partitions, target_partitions, counts);
//    if (!s.is_ok()) {
//      std::cout << "ERROR " << s.get_msg() << std::endl;
//      return 1;
//    }
//  }

//  cylon::RangePartitionKernel<arrow::Int64Type> kern(ctx,
//                                                     num_partitions,
//                                                     arrow_table->num_rows() * 0.1,
//                                                     num_partitions * 16);
//
//
//  kern.Partition(arrow_table->column(0), target_partitions, counts);
//
////  print_vec(target_partitions);
//  sleep(ctx->GetRank());
//  print_vec(counts);

  cylon::SortPartition(table,
                       idx_col,
                       num_partitions,
                       target_partitions,
                       counts,
                       true,
                       arrow_table->num_rows() * 0.1,
                       num_partitions * 1000);

//  std::vector<std::shared_ptr<arrow::Array>> out;
//  std::shared_ptr<cylon::ArrowArraySplitKernel> kern;
//
//  cylon::CreateSplitter(arrow::int64(), pool, &kern);
//
//  std::shared_ptr<arrow::ChunkedArray> col = arrow_table->column(2);
//  kern->Split(col, target_partitions, num_partitions, counts, out);

//  for (int i = 0; i < num_partitions; i++) {
//    std::cout << i << " count " << counts[i] << " out " << out[i]->length() << std::endl;
//  }

  std::vector<std::shared_ptr<cylon::Table>> output;

  cylon::Split(table, num_partitions, target_partitions, counts, output);

  std::cout << "original" << std::endl;
//  table->Print();

  if (ctx->GetRank() == 0) {
    for (int i = 0; i < num_partitions; i++) {
      std::cout << "partition " << i << " " << output[i]->Rows() << std::endl;
//    output[i]->Print();
    }
  }

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

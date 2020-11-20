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

#include <arrow/arrow_kernels.hpp>
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
  std::vector<uint32_t> target_partitions;
  target_partitions.reserve(arrow_table->num_rows());
  std::vector<uint32_t> counts;
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

  for (int i = 0; i < 4; i++) {
    target_partitions.clear();
    counts.clear();
    auto s = cylon::ModuloPartition(table, 0, num_partitions, target_partitions, counts);
    if (!s.is_ok()) return 1;
  }

//  print_vec(target_partitions);
//  print_vec(counts);


/*  std::vector<std::shared_ptr<arrow::Array>> out;
  std::shared_ptr<cylon::ArrowArraySplitKernel> kern;

  cylon::CreateSplitter(arrow::int64(), pool, &kern);

  std::shared_ptr<arrow::ChunkedArray> col = arrow_table->column(0);
  kern->Split(col, target_partitions, num_partitions, counts, out);

  for(int i = 0; i< num_partitions; i++){
    std::cout << i << " count " << counts[i] << " out " << out[i]->length() << std::endl;
  }*/

  std::vector<std::shared_ptr<cylon::Table>> output;

  cylon::Split(table, target_partitions, num_partitions, output, &counts);

  std::cout << "original" << std::endl;
//  table->Print();

  for (int i = 0; i < num_partitions; i++){
    std::cout << "partition " << i <<  " " << output[i]->Rows() << std::endl;
//    output[i]->Print();
  }

  ctx->Finalize();
  return 0;
}

void create_table(char *const *argv,
                  arrow::MemoryPool *pool,
                  std::shared_ptr<arrow::Table> &left_table) {
  arrow::Int64Builder left_id_builder(pool);
  arrow::DoubleBuilder cost_builder(pool);
  uint64_t count = std::stoull(argv[1]);
  double dup = std::stod(argv[2]);

  std::cout << "#### lines " << count << " dup " << dup << std::endl;

  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<int64_t> distrib(0, (int64_t) (count * dup));

  std::mt19937_64 gen1(rd());
  std::uniform_real_distribution<double> distrib1;

  arrow::Status st = left_id_builder.Reserve(count);
  st = cost_builder.Reserve(count);
  for (uint64_t i = 0; i < count; i++) {
    int64_t l = distrib(gen);
    double v = distrib1(gen1);
    left_id_builder.UnsafeAppend(l);
    cost_builder.UnsafeAppend(v);
  }

  std::shared_ptr<arrow::Array> left_id_array;
  std::shared_ptr<arrow::Array> cost_array;

  st = left_id_builder.Finish(&left_id_array);
  st = cost_builder.Finish(&cost_array);

  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
      arrow::field("first", arrow::int64()),
      arrow::field("second", arrow::float64())};
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  arrow::ArrayVector av1 = {left_id_array, left_id_array};
  arrow::ArrayVector av2 = {cost_array, cost_array};
  const std::shared_ptr<arrow::ChunkedArray> &ca1 = std::make_shared<arrow::ChunkedArray>(av1);
  const std::shared_ptr<arrow::ChunkedArray> &ca2 = std::make_shared<arrow::ChunkedArray>(av2);
  left_table = arrow::Table::Make(schema, {ca1, ca2});

}

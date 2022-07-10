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
#include <chrono>
#include <arrow/api.h>
#include <arrow/array.h>
#include <random>
#include <arrow/compute/api.h>

#include <cylon/net/mpi/mpi_communicator.hpp>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/util/builtins.hpp>
#include <cylon/table.hpp>
#include <cylon/groupby/groupby.hpp>

void create_table(char *const *argv,
                  arrow::MemoryPool *pool,
                  std::shared_ptr<arrow::Table> &left_table);

void CylonPipelineGroupBy(std::shared_ptr<cylon::Table> &ctable,
                          std::shared_ptr<cylon::Table> &output) {
  auto t1 = std::chrono::steady_clock::now();

  cylon::Status s =
      cylon::DistributedPipelineGroupBy(ctable, 0, {1}, {cylon::compute::SUM}, output);

  if (!s.is_ok()) {
    std::cout << " status " << s.get_code() << " " << s.get_msg() << std::endl;
    return;
  }

  auto t3 = std::chrono::steady_clock::now();
  std::cout << "hash_group3 " << output->Rows()
            << " " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t1).count()
            << " status " << s.get_code()
            << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    LOG(ERROR) << "There should be 2 args. count, duplication factor";
    return 1;
  }

  auto start_start = std::chrono::steady_clock::now();
  auto mpi_config = cylon::net::MPIConfig::Make();
    std::shared_ptr<cylon::CylonContext> ctx;
  if (!cylon::CylonContext::InitDistributed(mpi_config, &ctx).is_ok()) {
    std::cerr << "ctx init failed! " << std::endl;
    return 1;
  }

  arrow::MemoryPool *pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::Table> left_table;
//  create_binary_table(argv, ctx, pool, left_table, right_table);
  create_table(argv, pool, left_table);
  ctx->Barrier();

  std::shared_ptr<cylon::Table> first_table;
  auto status = cylon::Table::FromArrowTable(ctx, left_table, first_table);
  if (!status.is_ok()) {
    LOG(INFO) << "Table reading failed " << argv[1];
    ctx->Finalize();
    return 1;
  }

  auto read_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Read tables in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                read_end_time - start_start).count() << "[ms]";

  WriteCSV(first_table, "/tmp/source" + std::to_string(ctx->GetRank()) + ".txt");
//  first_table->Print();
  std::cout << "++++++++++++++++++++++++++" << std::endl;

  std::shared_ptr<cylon::Table> sorted_table;
  auto t1 = std::chrono::steady_clock::now();
  Sort(first_table, 0, sorted_table);
  auto t2 = std::chrono::steady_clock::now();

//  sorted_table->Print();
  std::cout << "++++++++++++++++++++++++++" << std::endl;

  LOG(INFO) << "sorted table in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                t2 - t1).count() << "[ms]";

  std::shared_ptr<cylon::Table> output;

  CylonPipelineGroupBy(sorted_table, output);
//  output->Print();
  WriteCSV(output, "/tmp/out" + std::to_string(ctx->GetRank()) + ".txt");
  output.reset();
  std::cout << "++++++++++++++++++++++++++" << std::endl;

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

  left_table = arrow::Table::Make(schema, {std::move(left_id_array), std::move(cost_array)});

}

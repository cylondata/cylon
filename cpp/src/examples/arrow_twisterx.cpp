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

#include <mpi.h>
#include <iostream>

#include <arrow/api.h>
#include <arrow/array/builder_primitive.h>
#include <glog/logging.h>
#include <net/mpi/mpi_communicator.h>

#include "net/ops/all_to_all.hpp"
#include "arrow/arrow_all_to_all.hpp"

using arrow::DoubleBuilder;
using arrow::Int64Builder;

class Clbk : public cylon::ArrowCallback {
 public:
  bool onReceive(int source, std::shared_ptr<arrow::Table> table) override {
    auto ids =
        std::static_pointer_cast<arrow::Int64Array>(table->column(0)->chunk(0));
    auto costs =
        std::static_pointer_cast<arrow::DoubleArray>(table->column(1)->chunk(0));
    for (int64_t i = 0; i < table->num_rows(); i++) {
      int64_t id = ids->Value(i);
      double cost = costs->Value(i);
      if (i % 100000 == 0) {
        LOG(INFO) << "ID " << id << " cost " << cost;
      }
    }
    return true;
  }
};

int main(int argc, char *argv[]) {
  std::cout << "First - ";

  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  int rank = ctx->GetRank();
  int size = ctx->GetWorldSize();

  std::cout << "First - " << rank << " size " << size << std::endl;
  std::vector<int> sources;
  std::vector<int> targets;
  for (int i = 0; i < size; i++) {
    sources.push_back(i);
    targets.push_back(i);
  }

  arrow::MemoryPool *pool = arrow::default_memory_pool();

  Int64Builder id_builder(pool);
  DoubleBuilder cost_builder(pool);

  for (int i = 0; i < 1000000; i++) {
    id_builder.Append(10 + i);
    cost_builder.Append(0.2 + i);
    if (i % 100000 == 0) {
      LOG(INFO) << "Appended " << i;
    }
  }

  std::shared_ptr<Clbk> clbk = std::make_shared<Clbk>();
  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
      arrow::field("id", arrow::int64()), arrow::field("cost", arrow::float64())};
  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  cylon::ArrowAllToAll all(ctx, sources, targets, 0, clbk, schema, pool);

  std::shared_ptr<arrow::Array> id_array;
  id_builder.Finish(&id_array);
  std::shared_ptr<arrow::Array> cost_array;
  cost_builder.Finish(&cost_array);

  std::shared_ptr<arrow::Table> ptr = arrow::Table::Make(schema, {id_array, cost_array});
  LOG(INFO) << "Insert ";
  all.insert(ptr, (rank + 1) % size);

  all.finish();
  while (!all.isComplete()) {
  }
  all.close();

  ctx->Finalize();
  return 0;
}

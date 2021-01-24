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

#include <ctx/cylon_context.hpp>
#include <net/mpi/mpi_communicator.hpp>
#include <table.hpp>
#include <ops/dis_union_op.hpp>

int main(int argc, char *argv[]) {

  auto mpi_config = cylon::net::MPIConfig::Make();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  std::shared_ptr<cylon::Table> table1, table2, out;
  cylon::FromCSV(ctx, argv[1], table1);
  cylon::FromCSV(ctx, argv[2], table2);

  std::shared_ptr<arrow::Table> table1_arr, table2_arr;
  table1->ToArrowTable(table1_arr);
  table2->ToArrowTable(table2_arr);

  LOG(INFO) << "read table";
  class Cb : public cylon::ResultsCallback {
   public:
    void OnResult(int tag, std::shared_ptr<cylon::Table> table) override {
      LOG(INFO) << "Result received " << table->Rows();
    }
  };

  std::shared_ptr<cylon::ResultsCallback> cb = std::make_shared<Cb>();

  std::shared_ptr<cylon::DisUnionOpConfig> union_config = std::make_shared<cylon::DisUnionOpConfig>();

  auto union_op = cylon::DisUnionOp(ctx, table1_arr->schema(), 0, cb,
                                    union_config);
  LOG(INFO) << "Created  op";

  LOG(INFO) << "Adding a table with " << table1->Rows();
  union_op.InsertTable(0, table1);

  LOG(INFO) << "Adding a table with " << table2->Rows();
  union_op.InsertTable(1, table2);

  auto execution = union_op.GetExecution();
  execution->WaitForCompletion();

  ctx->Finalize();
  return 0;
}

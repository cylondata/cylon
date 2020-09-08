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

  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  shared_ptr<cylon::Table> table1, table2, out;
  cylon::Table::FromCSV(ctx, argv[0], table1);
  cylon::Table::FromCSV(ctx, argv[1], table2);

  shared_ptr<arrow::Table> table1_arr, table2_arr;
  table1->ToArrowTable(table1_arr);
  table2->ToArrowTable(table2_arr);

  LOG(INFO) << "read table";
  class Cb : public cylon::ResultsCallback {
   public:
    virtual void OnResult(int tag, std::shared_ptr<cylon::Table> table) {
      LOG(INFO) << "Result received " << table->Rows();
    }
  };

  auto cb = std::make_shared<Cb>();

  auto union_config = std::make_shared<cylon::DisUnionOpConfig>();

  auto union_op = cylon::DisUnionOp(std::shared_ptr<cylon::CylonContext>(ctx),
                                    table1_arr->schema(), 0, cb, union_config);
  LOG(INFO) << "Created  op";

  LOG(INFO) << "Adding a table with "<< table1->Rows();
  union_op.InsertTable(0, table1);

  LOG(INFO) << "Adding a table with "<< table2->Rows();
  union_op.InsertTable(1, table2);

  auto execution = union_op.GetExecution();
  execution->WaitForCompletion();

  ctx->Finalize();
  return 0;
}

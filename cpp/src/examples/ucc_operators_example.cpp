/*
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/**
 * Temporary example, merge to ucc_example afterwards
 * run the example as follows:
 *  mpirun -n 4 bin/ucc_example
 */

#include <iostream>

#include <ucc/api/ucc.h>

#include <cylon/ctx/cylon_context.hpp>
#include <cylon/net/ucc/ucc_operations.hpp>
#include <cylon/net/ucx/ucx_communicator.hpp>
#include <cylon/table.hpp>
#include <cylon/scalar.hpp>

auto read_options = cylon::io::config::CSVReadOptions()
                        .UseThreads(false)
                        .BlockSize(1 << 30)
                        .WithDelimiter('\t');

cylon::Status readInputCsv(int i,
                           const std::shared_ptr<cylon::CylonContext>& ctx,
                           std::shared_ptr<cylon::Table>& table) {
  return cylon::FromCSV(ctx,
                 "/home/ky/cylon/data/input/csv1_" + std::to_string(i) + ".csv",
                 table, read_options);
}

void testTableAllgather(std::shared_ptr<cylon::CylonContext>& ctx) {
  int ws = ctx->GetWorldSize();
  if(ws > 4) {
    std::cout<<"table allgather test can only take 4 or less processes."<<std::endl;
    return;
  }
  std::shared_ptr<cylon::Table> table;
  std::vector<std::shared_ptr<cylon::Table>> out, original(ws);

  readInputCsv(ctx->GetRank(), ctx, table);

  for(int i = 0; i < ws; i++) {
    readInputCsv(i, ctx, original[i]);
  }

  ctx->GetCommunicator()->AllGather(table, &out);

  for(int i = 0; i < ws; i++) {
    bool result;
    cylon::Equals(out[i], original[i], result);
    if(!result) {
      std::cout<<"table allgather test failed at rank "<< ctx->GetRank() <<std::endl;
    }
  }
  std::cout << "table allgather test passed at rank " << ctx->GetRank() << std::endl;
}

void testColumnAllgather(std::shared_ptr<cylon::CylonContext>& ctx) {
  std::shared_ptr<cylon::Column> input;
  std::vector<std::shared_ptr<cylon::Column>> output;
  std::vector<int32_t> v(10);

  for (int i = 0; i < 10; i++) {
    v[i] = i + ctx->GetRank() * 10 + 1;
  }
  cylon::Column::FromVector(v, input);

  ctx->GetCommunicator()->Allgather(input, &output);

  for(int i = 0; i < ctx->GetWorldSize(); i++) {
    auto c = output[i];
    for (int j = 0; j < 10; j++) {
      int32_t result = std::static_pointer_cast<arrow::Int32Scalar>(
          c->data()->GetScalar(j).ValueOrDie())->value;

      if(result != i * 10 + j + 1) {
        std::cout<<"column gather test failed"<<std::endl;
        return;
      }
    }
  }
  std::cout << "column gather test passed at rank " << ctx->GetRank() << std::endl;
}

void testScalarAllgather(std::shared_ptr<cylon::CylonContext>& ctx) {
  std::shared_ptr<cylon::Column> column;
  std::vector<int32_t> v = {ctx->GetRank()};
  cylon::Column::FromVector(v, column);

  std::shared_ptr<arrow::Scalar> arrow_scalar = column->data()->GetScalar(0).ValueOrDie();
  std::shared_ptr<cylon::Scalar> scalar = cylon::Scalar::Make(arrow_scalar);

  std::shared_ptr<cylon::Column> out;
  ctx->GetCommunicator()->Allgather(scalar, &out);

  for(int i = 0; i < ctx->GetWorldSize(); i++) {
    int32_t result = std::static_pointer_cast<arrow::Int32Scalar>(out->data()->GetScalar(i).ValueOrDie())->value;
    if(result != i) {
      std::cout<<"scalar gather test failed at rank "<<ctx->GetRank()<<std::endl;
      return;
    }
  }
  std::cout<<"scalar gather test passed at rank "<<ctx->GetRank()<<std::endl;
}

void gather(std::shared_ptr<cylon::Table>& table,
               std::shared_ptr<cylon::CylonContext>& ctx) {
  std::vector<std::shared_ptr<cylon::Table>> out;
  auto status = ctx->GetCommunicator()->Gather(table, 0, 1, &out);
  std::cout<<status.get_msg()<<std::endl;
  if (ctx->GetRank() == 0) {
    std::cout<<"out size: "<<out.size()<<std::endl;
    for (auto out_table : out) {
      out_table->Print();
      // std::cout<<out_table->get_table()->num_rows()<<std::endl;
    }
  }
}

void testTableBcast(std::shared_ptr<cylon::CylonContext>& ctx) {
  std::shared_ptr<cylon::Table> table, out, original;
  if(ctx->GetRank() == 0) {
    readInputCsv(0, ctx, table);
  } else {
    readInputCsv(0, ctx, original);
  }
  ctx->GetCommunicator()->Bcast(&table, 0, ctx);

  if(ctx->GetRank() != 0) {
    bool result;
    cylon::Equals(table, original, result);
    if(!result) {
      std::cout<<"table bcast test failed at rank " << ctx->GetRank() <<std::endl;
      return;
    }
  } 
  std::cout<<"table bcast test passed at rank "<< ctx->GetRank() <<std::endl;
}

void testColumnAllReduce(std::shared_ptr<cylon::CylonContext>& ctx) {
  std::shared_ptr<cylon::Column> input, output;
  std::vector<int> v(10);
  int ws = ctx->GetWorldSize();

  for(int i = 0; i < 10; i++) {
    v[i] = i + ctx->GetRank() * 10;
  }
  cylon::Column::FromVector(v, input);

  ctx->GetCommunicator()->AllReduce(input, cylon::net::ReduceOp::SUM, &output);

  for (int i = 0; i < 10; i++) {
    auto result = std::static_pointer_cast<arrow::Int32Scalar>(output->data()->GetScalar(i).ValueOrDie())->value;
    if(result != ws * i + (ws - 1) * ws / 2 * 10) {
      std::cout << "column allreduce test failed at rank " << ctx->GetRank()
                << std::endl;
      return;
    }
  }
  std::cout << "column allreduce test passed at rank " << ctx->GetRank()
            << std::endl;
}

void testScalarAllReduce(std::shared_ptr<cylon::CylonContext>& ctx) {
  std::shared_ptr<cylon::Column> column;
  int ws = ctx->GetWorldSize();
  std::vector<int32_t> v = {ctx->GetRank() + 1};
  cylon::Column::FromVector(v, column);

  std::shared_ptr<arrow::Scalar> arrow_scalar =
      column->data()->GetScalar(0).ValueOrDie();
  std::shared_ptr<cylon::Scalar> scalar = cylon::Scalar::Make(arrow_scalar), out;

  ctx->GetCommunicator()->AllReduce(scalar, cylon::net::ReduceOp::SUM, &out);

  int32_t result = std::static_pointer_cast<arrow::Int32Scalar>(out->data())->value;
  
  if (result != (ws + 1) * ws / 2) {
    std::cout << "scalar allreduce test failed at rank " << ctx->GetRank()
              << std::endl;
    return;
  }
  std::cout << "scalar allreduce test passed at rank " << ctx->GetRank()
            << std::endl;
}

int main(int argc, char **argv) {
  auto ucx_config = std::make_shared<cylon::net::UCXConfig>();
  std::shared_ptr<cylon::CylonContext> ctx;
  if (!cylon::CylonContext::InitDistributed(ucx_config, &ctx).is_ok()) {
    std::cerr << "ctx init failed! " << std::endl;
    return 1;
  }

  testTableAllgather(ctx);
  testColumnAllgather(ctx);
  testScalarAllgather(ctx);
  testTableBcast(ctx);
  testColumnAllReduce(ctx);
  testScalarAllReduce(ctx);
}

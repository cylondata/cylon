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

#include <net/mpi/mpi_communicator.hpp>
#include <ctx/cylon_context.hpp>
#include <table.hpp>
#include <compute/aggregates.hpp>

cylon::Status CreateTable(std::shared_ptr<cylon::CylonContext> &ctx, int rows,
                          std::shared_ptr<cylon::Table> *output) {
  std::shared_ptr<std::vector<int32_t>> col0 = std::make_shared<std::vector<int32_t >>();
  std::shared_ptr<std::vector<double_t>> col1 = std::make_shared<std::vector<double_t >>();

  for (int i = 0; i < rows; i++) {
    col0->push_back((int32_t) i);
    col1->push_back((double_t) i + 10.0);
  }

  auto c0 = cylon::VectorColumn<int32_t>::Make("col0", cylon::Int32(), col0);
  auto c1 = cylon::VectorColumn<double_t>::Make("col1", cylon::Double(), col1);

  return cylon::Table::FromColumns(ctx, {c0, c1}, output);
}

int main() {
  cylon::Status status;
  const int rows = 4;
  int32_t agg_index = 1;

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  std::shared_ptr<cylon::Table> table;
  std::shared_ptr<cylon::compute::Result> result;

  status = CreateTable(ctx, rows, &table);

  if ((status.is_ok() && table->Columns() == 2 && table->Rows() == rows)) {
    table->Print();
  } else {
    std::cout << "table creation failed! " << status.get_msg() << std::endl;
    return 1;
  }

  if ((status = cylon::compute::Sum(table, agg_index, &result)).is_ok()) {
    const std::shared_ptr<arrow::DoubleScalar>
        &aa = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    std::cout << "sum " << aa->value << " " << status.get_code() << std::endl;
  } else {
    std::cout << "sum failed! " << status.get_msg() << std::endl;
  }

  if ((status = cylon::compute::Count(table, agg_index, &result)).is_ok()) {
    const auto &aa = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    std::cout << "count " << aa->value << " " << status.get_code() << std::endl;
  } else {
    std::cout << "count failed! " << status.get_msg() << std::endl;
  }

  if ((status = cylon::compute::Min(table, agg_index, &result)).is_ok()) {
    const auto &aa = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    std::cout << "min " << aa->value << " " << status.get_code() << std::endl;
  } else {
    std::cout << "min failed! " << status.get_msg() << std::endl;
  }

  if ((status = cylon::compute::Max(table, agg_index, &result)).is_ok()) {
    const std::shared_ptr<arrow::DoubleScalar>
        &aa = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    std::cout << "max " << aa->value << " " << status.get_code() << std::endl;
  } else {
    std::cout << "max failed! " << status.get_msg() << std::endl;
  }

  // Adding Python funcs
  std::shared_ptr<cylon::Table> output;

  if ((status = cylon::compute::Sum(table, agg_index, output)).is_ok()) {
    output->Print();
    auto array = output->GetColumn(0)->GetColumnData()->chunk(0);
    auto val = std::static_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(array)->Value(0);
    std::cout << "tsum:type " << array->type()->ToString() << ", value = " << val << " " << status.get_code() << std::endl;

  } else {
    std::cout << "Table: sum failed! " << status.get_msg() << std::endl;
  }

  if ((status = cylon::compute::Count(table, agg_index, output)).is_ok()) {
    output->Print();
  } else {
    std::cout << "Table: count failed! " << status.get_msg() << std::endl;
  }

  if ((status = cylon::compute::Max(table, agg_index, output)).is_ok()) {
    output->Print();
  } else {
    std::cout << "Table: max failed! " << status.get_msg() << std::endl;
  }

  if ((status = cylon::compute::Min(table, agg_index, output)).is_ok()) {
    output->Print();
  } else {
    std::cout << "Table: min failed! " << status.get_msg() << std::endl;
  }

  ctx->Finalize();
  return 0;
}
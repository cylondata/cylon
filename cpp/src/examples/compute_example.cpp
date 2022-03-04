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

#include <cylon/net/mpi/mpi_communicator.hpp>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/table.hpp>
#include <cylon/compute/aggregates.hpp>

cylon::Status CreateTable(const std::shared_ptr<cylon::CylonContext> &ctx, int rows,
                          std::shared_ptr<cylon::Table> &output) {
  std::vector<int32_t> col0;
  std::vector<double_t> col1;

  int rank = ctx->GetRank() + 1;
  for (int i = 0; i < rows; i++) {
    col0.push_back((int32_t) i * rank);
    col1.push_back((double_t) i * rank + 10.0);
  }

  std::shared_ptr<cylon::Column> c0, c1;
  RETURN_CYLON_STATUS_IF_FAILED(cylon::Column::FromVector(col0, c0));
  RETURN_CYLON_STATUS_IF_FAILED(cylon::Column::FromVector(col1, c1));

  return cylon::Table::FromColumns(ctx, {std::move(c0), std::move(c1)}, {"col0", "col1"}, output);
}

int main() {
  cylon::Status status;
  const int rows = 4;
  int32_t agg_index = 1;

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  std::shared_ptr<cylon::Table> table;
  std::shared_ptr<cylon::compute::Result> result;

  status = CreateTable(ctx, rows, table);

  if ((status.is_ok() && table->Columns() == 2 && table->Rows() == rows)) {
    table->Print();
  } else {
    std::cout << "table creation failed! " << status.get_msg() << std::endl;
    return 1;
  }

  if ((status = cylon::compute::Sum(table, agg_index, result)).is_ok()) {
    const std::shared_ptr<arrow::DoubleScalar>
        &aa = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    std::cout << "sum " << aa->value << " " << status.get_code() << std::endl;
  } else {
    std::cout << "sum failed! " << status.get_msg() << std::endl;
  }

  if ((status = cylon::compute::Count(table, agg_index, result)).is_ok()) {
    const auto &aa = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    std::cout << "count " << aa->value << " " << status.get_code() << std::endl;
  } else {
    std::cout << "count failed! " << status.get_msg() << std::endl;
  }

  if ((status = cylon::compute::Min(table, agg_index, result)).is_ok()) {
    const auto &aa = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    std::cout << "min " << aa->value << " " << status.get_code() << std::endl;
  } else {
    std::cout << "min failed! " << status.get_msg() << std::endl;
  }

  if ((status = cylon::compute::Max(table, agg_index, result)).is_ok()) {
    const std::shared_ptr<arrow::DoubleScalar>
        &aa = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    std::cout << "max " << aa->value << " " << status.get_code() << std::endl;
  } else {
    std::cout << "max failed! " << status.get_msg() << std::endl;
  }

  if ((status = cylon::compute::MinMax(table, agg_index, result)).is_ok()) {
    const auto &struct_scalar = result->GetResult().scalar_as<arrow::StructScalar>();

    const auto &aa = std::static_pointer_cast<arrow::DoubleScalar>(struct_scalar.value[0]);
    const auto &aa1 = std::static_pointer_cast<arrow::DoubleScalar>(struct_scalar.value[1]);
    std::cout << "min " << aa->value << " max " << aa1->value << " " << status.get_code()
              << std::endl;
  } else {
    std::cout << "minmax failed! " << status.get_msg() << std::endl;
  }

  // Adding Python funcs
  std::shared_ptr<cylon::Table> output;

  if ((status = cylon::compute::Sum(table, agg_index, output)).is_ok()) {
    output->Print();
    auto array = output->get_table()->column(0)->chunk(0);
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
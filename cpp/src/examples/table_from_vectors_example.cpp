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


/**
 * This example reads two csv files and does a union on them.
 */
int main() {
  cylon::Status status;
  const int size = 12;

  auto ctx = cylon::CylonContext::Init();

  std::shared_ptr<std::vector<int32_t>> col0 = std::make_shared<std::vector<int32_t>>();
  std::shared_ptr<std::vector<double_t>> col1 = std::make_shared<std::vector<double_t>>();

  for (int i = 0; i < size; i++) {
    col0->push_back(i);
    col1->push_back((double_t) i + 10.0);
  }

  auto cy_col0 = cylon::VectorColumn<int32_t>::Make("col0", cylon::Int32(), col0);
  auto cy_col1 = cylon::VectorColumn<double>::Make("col1", cylon::Double(), col1);

  std::shared_ptr<cylon::Table> output;
  status = cylon::Table::FromColumns(ctx, {cy_col0, cy_col1}, &output);

  if ((status.is_ok() && output->Columns() == 2 && output->Rows() == size)) {
    output->Print();
  }

  std::shared_ptr<arrow::DoubleArray>
      c = std::static_pointer_cast<arrow::DoubleArray>(output->GetColumn(1)->GetColumnData()->chunk(0));

  for (int i = 0; i < c->length(); i++) {
    std::cout << c->Value(i) << " ";
  }
  std::cout << std::endl;

  return 0;
}

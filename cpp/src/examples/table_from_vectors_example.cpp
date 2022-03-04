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

#include <cylon/net/mpi/mpi_communicator.hpp>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/table.hpp>


/**
 * This example reads two csv files and does a union on them.
 */
int main() {
  cylon::Status status;
  const int size = 10;

  auto ctx = cylon::CylonContext::Init();

  std::vector<int32_t> col0;
  std::vector<double_t> col1;

  for (int i = 0; i < size; i++) {
    col0.push_back(i);
    col1.push_back((double_t) i + 10.0);
  }

  std::shared_ptr<cylon::Column> c0, c1;
  status = cylon::Column::FromVector(col0, c0);
  if (!status.is_ok()) return 1;
  status = cylon::Column::FromVector(col1, c1);
  if (!status.is_ok()) return 1;

  std::shared_ptr<cylon::Table> output;
  status = cylon::Table::FromColumns(ctx, {std::move(c0), std::move(c1)}, {"col0", "col1"}, output);

  LOG(INFO) << "Read tables. Row Count: " << output->Rows();

  if ((status.is_ok() && output->Columns() == 2 && output->Rows() == size)) {
    output->Print();
  }

  auto c = std::static_pointer_cast<arrow::DoubleArray>(output->get_table()->column(1)->chunk(0));

  for (int i = 0; i < c->length(); i++) {
    std::cout << c->Value(i) << " ";
  }
  std::cout << std::endl;

  return 0;
}

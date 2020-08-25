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

#include "test_header.hpp"
#include "test_utils.hpp"

using namespace cylon;

cylon::Status CreateTable(int rows, std::shared_ptr<cylon::Table> *output) {
  std::shared_ptr<std::vector<int32_t>> col0 = std::make_shared<std::vector<int32_t >>();
  std::shared_ptr<std::vector<double_t>> col1 = std::make_shared<std::vector<double_t >>();

  for (int i = 0; i < rows; i++) {
    col0->push_back(i);
    col1->push_back((double_t) i + 10.0);
  }

  auto c0 = cylon::VectorColumn<int32_t>::Make("col0", cylon::Int32(), col0);
  auto c1 = cylon::VectorColumn<double>::Make("col1", cylon::Double(), col1);

  return cylon::Table::FromColumns(ctx, {c0, c1}, output);
}

TEST_CASE("create table from columns testing", "[columns]") {
  cylon::Status status;
  const int size = 12;

  SECTION("testing create table") {
    std::shared_ptr<cylon::Table> output;
    status = CreateTable(size, &output);

    REQUIRE((status.is_ok() && output->Columns() == 2 && output->Rows() == size));

    std::shared_ptr<arrow::DoubleArray> c =
        static_pointer_cast<arrow::DoubleArray>(output->GetColumn(1)->GetColumnData()->chunk(0));

    for (int i = 0; i < c->length(); i++) {
      REQUIRE((c->Value(i) == i + 10.0));
    }
  }
}
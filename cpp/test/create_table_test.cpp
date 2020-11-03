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

TEST_CASE("create table from columns testing", "[columns]") {
  cylon::Status status;
  const int size = 12;

  SECTION("testing create table") {
    std::shared_ptr<cylon::Table> output;
    status = cylon::test::CreateTable(ctx, size, output);

    REQUIRE((status.is_ok() && output->Columns() == 2 && output->Rows() == size));

    std::shared_ptr<arrow::DoubleArray> c =
        std::static_pointer_cast<arrow::DoubleArray>(output->GetColumn(1)->GetColumnData()->chunk(0));

    for (int i = 0; i < c->length(); i++) {
      REQUIRE((c->Value(i) == i + 10.0));
    }
  }
}
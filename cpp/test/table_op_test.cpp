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

#include <compute/aggregates.hpp>

using namespace cylon;

TEST_CASE("table ops testing", "[table_ops]") {
  cylon::Status status;
  const int size = 12;
  std::shared_ptr<cylon::Table> input, select;
  status = cylon::test::CreateTable(ctx, size, input);

  SECTION("table creation") {
    REQUIRE((status.is_ok() && input->Columns() == 2 && input->Rows() == size));
  }

  SECTION("testing select") {
    status = Select(input, [](cylon::Row row) {
      return row.GetInt32(0) % 2 == 0;
    }, select);

    REQUIRE((status.is_ok() && select->Columns() == 2 && select->Rows() == size / 2));
  }

  SECTION("testing shuffle") {
    status = Shuffle(input, {0}, select);
    REQUIRE((status.is_ok() && select->Columns() == 2));

    std::shared_ptr<compute::Result> result;
    status = compute::Count(select, 0, result);
    auto s = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    REQUIRE((status.is_ok() && s->value == size * WORLD_SZ));
  }

}

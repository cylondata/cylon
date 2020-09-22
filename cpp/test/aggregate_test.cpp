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

#include <compute/aggregates.hpp>
#include "test_header.hpp"
#include "test_utils.hpp"

using namespace cylon;

TEST_CASE("aggregate testing", "[join]") {
  const int rows = 12;

  cylon::Status status;
  std::shared_ptr<cylon::Table> table;
  std::shared_ptr<cylon::compute::Result> result;
  status = cylon::test::CreateTable(ctx, rows, &table);

  SECTION("table creation") {
    REQUIRE((status.is_ok() && table->Columns() == 2 && table->Rows() == rows));
  }

  SECTION("testing sum") {
    status = cylon::compute::Sum(table, 1, &result);
    REQUIRE(status.is_ok());

    auto res_scalar = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    std::cout << " " << res_scalar->value << std::endl;
    REQUIRE(res_scalar->value == ((double) (rows * (rows - 1) / 2.0) + 10.0 * rows) * ctx->GetWorldSize());
  }

  SECTION("testing count") {
    status = cylon::compute::Count(table, 1, &result);
    REQUIRE(status.is_ok());

    auto res_scalar = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    REQUIRE(res_scalar->value == rows * ctx->GetWorldSize());
  }

  SECTION("testing min") {
    status = cylon::compute::Min(table, 1, &result);
    REQUIRE(status.is_ok());

    auto res_scalar = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    REQUIRE(res_scalar->value == 10.0);
  }

  SECTION("testing max") {
    status = cylon::compute::Max(table, 1, &result);
    REQUIRE(status.is_ok());

    auto res_scalar = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    REQUIRE(res_scalar->value == 10.0 + (double) (rows - 1));
  }
}

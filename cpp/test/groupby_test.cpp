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

#include <cylon/net/mpi/mpi_communicator.hpp>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/util/builtins.hpp>
#include <cylon/table.hpp>
#include <cylon/util/arrow_utils.hpp>
#include <cylon/groupby/groupby.hpp>
#include <cylon/compute/aggregates.hpp>

#include "common/test_header.hpp"

namespace cylon {
namespace test {

Status create_table(const std::shared_ptr<CylonContext> &ctx_,
                    const std::shared_ptr<arrow::DataType> &value_type,
                    std::shared_ptr<Table> &table) {
  auto col0 = ArrayFromJSON(arrow::int64(), "[0, 0, 1, 1, 2, 2, 3, 3]");
  auto col1 = ArrayFromJSON(value_type, "[0, 0, 1, 1, 2, 2, 3, 3]");

  auto schema = arrow::schema({field("col0", arrow::int64()), field("col1", value_type)});
  auto atable = arrow::Table::Make(std::move(schema), {std::move(col0), std::move(col1)});

  return Table::FromArrowTable(ctx_, std::move(atable), table);
}

Status HashCylonGroupBy(std::shared_ptr<Table> &ctable,
                        const compute::AggregationOpId &aggregate_ops,
                        std::shared_ptr<Table> &output) {

  CHECK_CYLON_STATUS(DistributedHashGroupBy(ctable, 0, {1}, {aggregate_ops}, output));
  INFO("hash_group op:" << aggregate_ops << " rows:" << output->Rows());
  return Status::OK();
}

Status PipelineCylonGroupBy(std::shared_ptr<Table> &ctable,
                            std::shared_ptr<Table> &output) {

  CHECK_CYLON_STATUS(DistributedPipelineGroupBy(ctable, 0, {1}, {compute::SUM}, output));
  INFO("pipe_group " << output->Rows());
  return Status::OK();
}

TEMPLATE_LIST_TEST_CASE("groupby testing", "[groupby]", ArrowNumericTypes) {
  auto val_type = default_type_instance<TestType>();

  using ValScalarT = typename arrow::TypeTraits<TestType>::ScalarType;
  using T = typename TestType::c_type;

  INFO("Testing groupby - " + val_type->ToString());

  std::shared_ptr<Table> table, output1, output2, validate;
  CHECK_CYLON_STATUS(create_table(ctx, val_type, table));

  std::shared_ptr<compute::Result> result;

  SECTION("testing hash group by result") {
    CHECK_CYLON_STATUS(HashCylonGroupBy(table, compute::SUM, output1));

    CHECK_CYLON_STATUS(compute::Sum(output1, 0, result));
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    INFO("idx_sum " << idx_sum->value);
    REQUIRE(idx_sum->value == 6); // 3* 4/ 2

    CHECK_CYLON_STATUS(compute::Sum(output1, 1, result));
    auto val_sum = std::static_pointer_cast<ValScalarT>(result->GetResult().scalar());
    INFO("val_sum " << val_sum->ToString());
    REQUIRE(val_sum->value == (T) (2 * 6 * ctx->GetWorldSize()));
  }

  SECTION("testing hash group by count") {
    CHECK_CYLON_STATUS(HashCylonGroupBy(table, compute::COUNT, output1));

    CHECK_CYLON_STATUS(compute::Sum(output1, 0, result));
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    INFO("idx_sum " << idx_sum->value);
    REQUIRE(idx_sum->value == 6); // 3* 4/ 2

    CHECK_CYLON_STATUS(compute::Sum(output1, 1, result));
    auto val_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    INFO("val_sum " << val_sum->value);
    REQUIRE(val_sum->value == 4 * 2 * ctx->GetWorldSize());
  }

  SECTION("testing hash group by mean") {
    CHECK_CYLON_STATUS(HashCylonGroupBy(table, compute::MEAN, output1));

    CHECK_CYLON_STATUS(compute::Sum(output1, 0, result));
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    INFO("idx_sum " << idx_sum->value);
    REQUIRE(idx_sum->value == 6); // 3* 4/ 2

    CHECK_CYLON_STATUS(compute::Sum(output1, 1, result));
    auto val_sum = std::static_pointer_cast<ValScalarT>(result->GetResult().scalar());
    INFO("val_sum " << val_sum->value);
    REQUIRE(val_sum->value == 6.0);
  }

  SECTION("testing hash group by var") {
    CHECK_CYLON_STATUS(HashCylonGroupBy(table, compute::VAR, output1));

    CHECK_CYLON_STATUS(compute::Sum(output1, 0, result));
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    INFO("idx_sum " << idx_sum->value);
    REQUIRE(idx_sum->value == 6); // 3* 4/ 2

    CHECK_CYLON_STATUS(compute::Sum(output1, 1, result));
    auto val_sum = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    INFO("val_sum " << val_sum->value);
    REQUIRE(val_sum->value == 0.0);
  }

  SECTION("testing hash group by stddev") {
    CHECK_CYLON_STATUS(HashCylonGroupBy(table, compute::STDDEV, output1));

    CHECK_CYLON_STATUS(compute::Sum(output1, 0, result));
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    INFO("idx_sum " << idx_sum->value);
    REQUIRE(idx_sum->value == 6); // 3* 4/ 2

    CHECK_CYLON_STATUS(compute::Sum(output1, 1, result));
    auto val_sum = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    INFO("val_sum " << val_sum->value);
    REQUIRE(val_sum->value == 0.0);
  }

  SECTION("testing hash group by nunique") {
    CHECK_CYLON_STATUS(HashCylonGroupBy(table, compute::NUNIQUE, output1));

    CHECK_CYLON_STATUS(compute::Sum(output1, 0, result));
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    INFO("idx_sum " << idx_sum->value);
    REQUIRE(idx_sum->value == 6); // 3* 4/ 2

    CHECK_CYLON_STATUS(compute::Sum(output1, 1, result));
    auto val_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    INFO("val_sum " << val_sum->value);
    REQUIRE(val_sum->value == 1 * 4);
  }

  SECTION("testing hash group by median") {
    CHECK_CYLON_STATUS(HashCylonGroupBy(table, compute::QUANTILE, output1));

    CHECK_CYLON_STATUS(compute::Sum(output1, 0, result));
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    INFO("idx_sum " << idx_sum->value);
    REQUIRE(idx_sum->value == 6); // 3* 4/ 2

    CHECK_CYLON_STATUS(compute::Sum(output1, 1, result));
    auto val_sum = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    INFO("val_sum " << val_sum->value);
    REQUIRE(val_sum->value == (T) 6); // 0 + .. + 3
  }

  SECTION("testing pipeline group by") {
    CHECK_CYLON_STATUS(Sort(table, 0, output1));

    CHECK_CYLON_STATUS(PipelineCylonGroupBy(output1, output2));

    CHECK_CYLON_STATUS(compute::Sum(output2, 0, result));
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    INFO("idx_sum " << idx_sum->value);
    REQUIRE(idx_sum->value == 6); // 3* 4/ 2

    CHECK_CYLON_STATUS(compute::Sum(output2, 1, result));
    auto val_sum = std::static_pointer_cast<ValScalarT>(result->GetResult().scalar());
    INFO("val_sum " << val_sum->value);
    REQUIRE(val_sum->value == (T) (2 * 6 * ctx->GetWorldSize()));
  }
}

} // namespace test 
} // namespace cylon



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

#include <random>

#include <cylon/net/mpi/mpi_communicator.hpp>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/util/builtins.hpp>
#include <cylon/table.hpp>
#include <cylon/util/arrow_utils.hpp>
#include <cylon/groupby/groupby.hpp>
#include <cylon/compute/aggregates.hpp>

#include "common/test_header.hpp"
#include "test_macros.hpp"

namespace cylon {
namespace test {

Status create_table(const std::shared_ptr<CylonContext> &ctx_, std::shared_ptr<Table> &table) {
  std::vector<int64_t> col0{0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
  std::vector<double> col1{0, 0, 1, 1, 2, 2, 3, 3, 4, 4};

  std::shared_ptr<Column> c0, c1;
  RETURN_CYLON_STATUS_IF_FAILED(Column::FromVector(ctx_, "col0", Int64(), col0, c0));
  RETURN_CYLON_STATUS_IF_FAILED(Column::FromVector(ctx_, "col1", Double(), col1, c1));

  return Table::FromColumns(ctx_, {std::move(c0), std::move(c1)}, table);
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

TEST_CASE("groupby testing", "[groupby]") {
  INFO("Testing groupby");

  std::shared_ptr<Table> table, output1, output2, validate;
  CHECK_CYLON_STATUS(create_table(ctx, table));

  REQUIRE((table->Columns() == 2 && table->Rows() == 10));

  std::shared_ptr<compute::Result> result;

  SECTION("testing hash group by result") {
    CHECK_CYLON_STATUS(HashCylonGroupBy(table, compute::SUM, output1));

    CHECK_CYLON_STATUS(compute::Sum(output1, 0, result));
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    INFO("idx_sum " << idx_sum->value);
    REQUIRE(idx_sum->value == 10); // 4* 5/ 2

    CHECK_CYLON_STATUS(compute::Sum(output1, 1, result));
    auto val_sum = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    INFO("val_sum " << val_sum->value);
    REQUIRE(val_sum->value == 2 * 10.0 * ctx->GetWorldSize());
  }

  SECTION("testing hash group by count") {
    CHECK_CYLON_STATUS(HashCylonGroupBy(table, compute::COUNT, output1));

    CHECK_CYLON_STATUS(compute::Sum(output1, 0, result));
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    INFO("idx_sum " << idx_sum->value);
    REQUIRE(idx_sum->value == 10); // 4* 5/ 2

    CHECK_CYLON_STATUS(compute::Sum(output1, 1, result));
    auto val_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    INFO("val_sum " << val_sum->value);
    REQUIRE(val_sum->value == 5 * 2 * ctx->GetWorldSize());
  }

  SECTION("testing hash group by mean") {
    CHECK_CYLON_STATUS(HashCylonGroupBy(table, compute::MEAN, output1));

    CHECK_CYLON_STATUS(compute::Sum(output1, 0, result));
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    INFO("idx_sum " << idx_sum->value);
    REQUIRE(idx_sum->value == 10); // 4* 5/ 2

    CHECK_CYLON_STATUS(compute::Sum(output1, 1, result));
    auto val_sum = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    INFO("val_sum " << val_sum->value);
    REQUIRE(val_sum->value == 10.0);
  }

  SECTION("testing hash group by var") {
    CHECK_CYLON_STATUS(HashCylonGroupBy(table, compute::VAR, output1));

    CHECK_CYLON_STATUS(compute::Sum(output1, 0, result));
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    INFO("idx_sum " << idx_sum->value);
    REQUIRE(idx_sum->value == 10); // 4* 5/ 2

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
    REQUIRE(idx_sum->value == 10); // 4* 5/ 2

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
    REQUIRE(idx_sum->value == 10); // 4* 5/ 2

    CHECK_CYLON_STATUS(compute::Sum(output1, 1, result));
    auto val_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    INFO("val_sum " << val_sum->value);
    REQUIRE(val_sum->value == 1 * 5);
  }

  SECTION("testing hash group by median") {
    CHECK_CYLON_STATUS(HashCylonGroupBy(table, compute::QUANTILE, output1));

    CHECK_CYLON_STATUS(compute::Sum(output1, 0, result));
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    INFO("idx_sum " << idx_sum->value);
    REQUIRE(idx_sum->value == 10); // 4* 5/ 2

    CHECK_CYLON_STATUS(compute::Sum(output1, 1, result));
    auto val_sum = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    INFO("val_sum " << val_sum->value);
    REQUIRE(val_sum->value == 10); // 0 + .. + 4
  }

  SECTION("testing pipeline group by") {
    CHECK_CYLON_STATUS(Sort(table, 0, output1));

    CHECK_CYLON_STATUS(PipelineCylonGroupBy(output1, output2));

    CHECK_CYLON_STATUS(compute::Sum(output2, 0, result));
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    INFO("idx_sum " << idx_sum->value);
    REQUIRE(idx_sum->value == 10); // 4* 5/ 2

    CHECK_CYLON_STATUS(compute::Sum(output2, 1, result));
    auto val_sum = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    INFO("val_sum " << val_sum->value);
    REQUIRE(val_sum->value == 2 * 10.0 * ctx->GetWorldSize());
  }
}

} // namespace test 
} // namespace cylon



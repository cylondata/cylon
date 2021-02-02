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
#include <net/mpi/mpi_communicator.hpp>
#include <ctx/cylon_context.hpp>
#include <util/builtins.hpp>
#include <table.hpp>
#include <chrono>
#include <random>


#include <util/arrow_utils.hpp>
#include <groupby/groupby.hpp>
#include <compute/aggregates.hpp>

#include "test_header.hpp"

Status create_table(std::shared_ptr<cylon::Table> &table) {
  std::vector<int64_t> col0{0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
  std::vector<double> col1{0, 0, 1, 1, 2, 2, 3, 3, 4, 4};

  auto c0 = cylon::VectorColumn<int64_t>::Make("col0", cylon::Int64(), std::make_shared<std::vector<int64_t>>(col0));
  auto c1 = cylon::VectorColumn<double>::Make("col1", cylon::Double(), std::make_shared<std::vector<double>>(col1));
  return cylon::Table::FromColumns(ctx, {c0, c1}, table);
}

Status HashCylonGroupBy(std::shared_ptr<cylon::Table> &ctable,
                        const compute::AggregationOpId &aggregate_ops,
                        std::shared_ptr<cylon::Table> &output) {

  cylon::Status s =
      cylon::DistributedHashGroupBy(ctable, 0, {1}, {aggregate_ops}, output);

  LOG(INFO) << "hash_group op:" << aggregate_ops << " rows:" << output->Rows();
  return s;
}

Status PipelineCylonGroupBy(std::shared_ptr<cylon::Table> &ctable,
                            std::shared_ptr<cylon::Table> &output) {

  cylon::Status s =
      cylon::DistributedPipelineGroupBy(ctable, 0, {1}, {cylon::compute::SUM}, output);

  LOG(INFO) << "pipe_group " << output->Rows();
  return s;
}

TEST_CASE("groupby testing", "[groupby]") {
  LOG(INFO) << "Testing groupby";

  std::shared_ptr<cylon::Table> table, output1, output2, validate;
  auto status = create_table(table);

  REQUIRE((status.is_ok() && table->Columns() == 2 && table->Rows() == 10));

  std::shared_ptr<cylon::compute::Result> result;

  SECTION("testing hash group by result") {
    status = HashCylonGroupBy(table, cylon::compute::SUM, output1);
    REQUIRE(status.is_ok());

    status = cylon::compute::Sum(output1, 0, result);
    REQUIRE(status.is_ok());
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    std::cout << "idx_sum " << idx_sum->value << std::endl;
    REQUIRE(idx_sum->value == 10); // 4* 5/ 2

    status = cylon::compute::Sum(output1, 1, result);
    REQUIRE(status.is_ok());
    auto val_sum = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    std::cout << "val_sum " << val_sum->value << std::endl;
    REQUIRE(val_sum->value == 2 * 10.0 * ctx->GetWorldSize());
  }

  SECTION("testing hash group by count") {
    status = HashCylonGroupBy(table, cylon::compute::COUNT, output1);
    REQUIRE(status.is_ok());

    status = cylon::compute::Sum(output1, 0, result);
    REQUIRE(status.is_ok());
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    std::cout << "idx_sum " << idx_sum->value << std::endl;
    REQUIRE(idx_sum->value == 10); // 4* 5/ 2

    status = cylon::compute::Sum(output1, 1, result);
    REQUIRE(status.is_ok());
    auto val_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    std::cout << "val_sum " << val_sum->value << std::endl;
    REQUIRE(val_sum->value == 5 * 2 * ctx->GetWorldSize());
  }

  SECTION("testing hash group by mean") {
    status = HashCylonGroupBy(table, cylon::compute::MEAN, output1);
    REQUIRE(status.is_ok());

    status = cylon::compute::Sum(output1, 0, result);
    REQUIRE(status.is_ok());
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    std::cout << "idx_sum " << idx_sum->value << std::endl;
    REQUIRE(idx_sum->value == 10); // 4* 5/ 2

    status = cylon::compute::Sum(output1, 1, result);
    REQUIRE(status.is_ok());
    auto val_sum = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    std::cout << "val_sum " << val_sum->value << std::endl;
    REQUIRE(val_sum->value == 10.0);
  }

  SECTION("testing hash group by var") {
    status = HashCylonGroupBy(table, cylon::compute::VAR, output1);
    REQUIRE(status.is_ok());

    status = cylon::compute::Sum(output1, 0, result);
    REQUIRE(status.is_ok());
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    std::cout << "idx_sum " << idx_sum->value << std::endl;
    REQUIRE(idx_sum->value == 10); // 4* 5/ 2

    status = cylon::compute::Sum(output1, 1, result);
    REQUIRE(status.is_ok());
    auto val_sum = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    std::cout << "val_sum " << val_sum->value << std::endl;
    REQUIRE(val_sum->value == 0.0);
  }

  SECTION("testing hash group by nunique") {
    status = HashCylonGroupBy(table, cylon::compute::NUNIQUE, output1);
    REQUIRE(status.is_ok());

    status = cylon::compute::Sum(output1, 0, result);
    REQUIRE(status.is_ok());
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    std::cout << "idx_sum " << idx_sum->value << std::endl;
    REQUIRE(idx_sum->value == 10); // 4* 5/ 2

    status = cylon::compute::Sum(output1, 1, result);
    REQUIRE(status.is_ok());
    auto val_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    std::cout << "val_sum " << val_sum->value << std::endl;
    REQUIRE(val_sum->value == 1 * 5);
  }

  SECTION("testing hash group by median") {
    status = HashCylonGroupBy(table, cylon::compute::QUANTILE, output1);
    REQUIRE(status.is_ok());

    status = cylon::compute::Sum(output1, 0, result);
    REQUIRE(status.is_ok());
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    std::cout << "idx_sum " << idx_sum->value << std::endl;
    REQUIRE(idx_sum->value == 10); // 4* 5/ 2

    status = cylon::compute::Sum(output1, 1, result);
    REQUIRE(status.is_ok());
    auto val_sum = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    std::cout << "val_sum " << val_sum->value << std::endl;
    REQUIRE(val_sum->value == 10); // 0 + .. + 4
  }

  SECTION("testing pipeline group by") {
    status = Sort(table, 0, output1);
    REQUIRE(status.is_ok());

    status = PipelineCylonGroupBy(output1, output2);
    REQUIRE(status.is_ok());

    status = cylon::compute::Sum(output2, 0, result);
    REQUIRE(status.is_ok());
    auto idx_sum = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    std::cout << "idx_sum " << idx_sum->value << std::endl;
    REQUIRE(idx_sum->value == 10); // 4* 5/ 2

    status = cylon::compute::Sum(output2, 1, result);
    REQUIRE(status.is_ok());
    auto val_sum = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    std::cout << "val_sum " << val_sum->value << std::endl;
    REQUIRE(val_sum->value == 2*10.0* ctx->GetWorldSize());
  }

}



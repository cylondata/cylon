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

#include "cylon/compute/aggregates.hpp"
#include "cylon/mapreduce/mapreduce.hpp"
#include "cylon/net/mpi/mpi_operations.hpp"

#include "common/test_header.hpp"
#include "test_utils.hpp"

namespace cylon {
namespace test {

TEST_CASE("aggregate testing", "[aggregates]") {
  const int rows = 12;

  cylon::Status status;
  std::shared_ptr<cylon::Table> table;
  std::shared_ptr<cylon::Table> output;
  std::shared_ptr<cylon::compute::Result> result;
  status = cylon::test::CreateTable(ctx, rows, table);

  SECTION("table creation") {
    REQUIRE((status.is_ok() && table->Columns() == 2 && table->Rows() == rows));
  }

  SECTION("testing sum") {
    status = cylon::compute::Sum(table, 1, result);
    REQUIRE(status.is_ok());

    auto res_scalar = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    std::cout << " " << res_scalar->value << std::endl;
    REQUIRE(res_scalar->value
                == ((double) (rows * (rows - 1) / 2.0) + 10.0 * rows) * ctx->GetWorldSize());
  }

  SECTION("testing count") {
    status = cylon::compute::Count(table, 1, result);
    REQUIRE(status.is_ok());

    auto res_scalar = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
    REQUIRE(res_scalar->value == rows * ctx->GetWorldSize());
  }

  SECTION("testing min") {
    status = cylon::compute::Min(table, 1, result);
    REQUIRE(status.is_ok());

    auto res_scalar = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    REQUIRE(res_scalar->value == 10.0);
  }

  SECTION("testing max") {
    status = cylon::compute::Max(table, 1, result);
    REQUIRE(status.is_ok());

    auto res_scalar = std::static_pointer_cast<arrow::DoubleScalar>(result->GetResult().scalar());
    REQUIRE(res_scalar->value == 10.0 + (double) (rows - 1));
  }

    // Adding Table output based Aggregates

  SECTION("testing table:sum") {
    status = cylon::compute::Sum(table, 1, output);
    REQUIRE(status.is_ok());

    auto array = output->get_table()->column(0)->chunk(0);
    auto val = std::static_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(array)->Value(0);

    REQUIRE(val == ((double) (rows * (rows - 1) / 2.0) + 10.0 * rows) * ctx->GetWorldSize());
  }

  SECTION("testing table:count") {
    status = cylon::compute::Count(table, 1, output);
    REQUIRE(status.is_ok());

    auto array = output->get_table()->column(0)->chunk(0);
    auto val = std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(array)->Value(0);

    REQUIRE(val == rows * ctx->GetWorldSize());
  }

  SECTION("testing table:min") {
    status = cylon::compute::Min(table, 1, output);
    REQUIRE(status.is_ok());

    auto array = output->get_table()->column(0)->chunk(0);
    auto val = std::static_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(array)->Value(0);

    REQUIRE(val == 10.0);
  }

  SECTION("testing max") {
    status = cylon::compute::Max(table, 1, output);
    REQUIRE(status.is_ok());

    auto array = output->get_table()->column(0)->chunk(0);
    auto val = std::static_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(array)->Value(0);

    REQUIRE(val == 10.0 + (double) (rows - 1));
  }
}

TEMPLATE_LIST_TEST_CASE("mapred kernels", "[mapred]", ArrowNumericTypes) {
  auto type = default_type_instance<TestType>();
  auto pool = ToArrowPool(ctx);

  SECTION("sum") {
    INFO("sum " + type->ToString())
    auto kern = mapred::MakeMapReduceKernel(type, compute::SUM);

    // init
    kern->Init(pool, /*options=*/nullptr);

    // combine
    auto arr = ArrayFromJSON(type, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");
    auto g_ids = ArrayFromJSON(arrow::int64(), "[0, 0, 1, 1, 2, 2, 3, 3, 4, 4]");
    arrow::ArrayVector array_vector;
    CHECK_CYLON_STATUS(kern->CombineLocally(arr, g_ids, 5, &array_vector));
    REQUIRE(array_vector.size() == 1);
    REQUIRE(array_vector[0]->length() == 5);
    CHECK_ARROW_EQUAL(ArrayFromJSON(type, "[1, 5, 9, 13, 17]"), array_vector[0]);

    // reduce
    CHECK_CYLON_STATUS(kern->ReduceShuffledResults({arr}, g_ids, nullptr, 5, &array_vector));
    CHECK_ARROW_EQUAL(ArrayFromJSON(type, "[1, 5, 9, 13, 17]"), array_vector[0]);

    //finalize
    std::shared_ptr<arrow::Array> out;
    CHECK_CYLON_STATUS(kern->Finalize(array_vector, &out));
    CHECK_ARROW_EQUAL(array_vector[0], out);
  }

  SECTION("count") {
    INFO("count " + type->ToString())
    auto kern = mapred::MakeMapReduceKernel(type, compute::COUNT);

    // init
    kern->Init(pool, /*options=*/nullptr);

    // combine
    auto arr = ArrayFromJSON(type, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");
    auto g_ids = ArrayFromJSON(arrow::int64(), "[0, 0, 1, 1, 2, 2, 3, 3, 4, 4]");
    arrow::ArrayVector array_vector;
    CHECK_CYLON_STATUS(kern->CombineLocally(arr, g_ids, 5, &array_vector));
    REQUIRE(array_vector.size() == 1);
    REQUIRE(array_vector[0]->length() == 5);
    CHECK_ARROW_EQUAL(ArrayFromJSON(arrow::int64(), "[2, 2, 2, 2, 2]"), array_vector[0]);

    // reduce
    arr = ArrayFromJSON(arrow::int64(), "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");
    CHECK_CYLON_STATUS(kern->ReduceShuffledResults({arr}, g_ids, nullptr, 5, &array_vector));
    CHECK_ARROW_EQUAL(ArrayFromJSON(arrow::int64(), "[1, 5, 9, 13, 17]"), array_vector[0]);

    //finalize
    std::shared_ptr<arrow::Array> out;
    CHECK_CYLON_STATUS(kern->Finalize(array_vector, &out));
    CHECK_ARROW_EQUAL(array_vector[0], out);
  }

  SECTION("mean") {
    INFO("mean " + type->ToString())
    auto kern = mapred::MakeMapReduceKernel(type, compute::MEAN);

    // init
    kern->Init(pool, /*options=*/nullptr);

    // combine
    auto arr = ArrayFromJSON(type, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");
    auto g_ids = ArrayFromJSON(arrow::int64(), "[0, 0, 1, 1, 2, 2, 3, 3, 4, 4]");
    arrow::ArrayVector array_vector;
    CHECK_CYLON_STATUS(kern->CombineLocally(arr, g_ids, 5, &array_vector));
    REQUIRE(array_vector.size() == 2);
    REQUIRE(array_vector[0]->length() == 5);
    CHECK_ARROW_EQUAL(ArrayFromJSON(type, "[1, 5, 9, 13, 17]"), array_vector[0]);
    CHECK_ARROW_EQUAL(ArrayFromJSON(arrow::int64(), "[2, 2, 2, 2, 2]"), array_vector[1]);

    // reduce
    auto cnts = ArrayFromJSON(arrow::int64(), "[2, 3, 2, 3, 2, 3, 2, 3, 2, 3]");
    CHECK_CYLON_STATUS(
        kern->ReduceShuffledResults({arr, cnts}, g_ids, nullptr, 5, &array_vector));
    CHECK_ARROW_EQUAL(ArrayFromJSON(type, "[1, 5, 9, 13, 17]"), array_vector[0]);
    CHECK_ARROW_EQUAL(ArrayFromJSON(arrow::int64(), "[5, 5, 5, 5, 5]"), array_vector[1]);

    //finalize
    std::shared_ptr<arrow::Array> out;
    CHECK_CYLON_STATUS(kern->Finalize(array_vector, &out));
    auto exp = ArrayFromJSON(arrow::float64(), "[0.2, 1, 1.8, 2.6, 3.4]");
    auto exp_cast = *arrow::compute::Cast(*exp, type, arrow::compute::CastOptions::Unsafe());
    CHECK_ARROW_EQUAL(exp_cast, out);
  }
}

TEMPLATE_LIST_TEST_CASE("mapred local aggregate", "[mapred]", ArrowNumericTypes) {
  // if distributed, return
  if (ctx->GetWorldSize() > 1) { return; }

  auto type = default_type_instance<TestType>();

  auto key = ArrayFromJSON(type, "[0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4]");
  auto val = ArrayFromJSON(type, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]");

  auto schema = arrow::schema({arrow::field("a", type), arrow::field("b", type)});
  std::shared_ptr<Table> table, output;
  CHECK_CYLON_STATUS(Table::FromArrowTable(ctx, arrow::Table::Make(schema, {key, val}), table));

  mapred::AggOpVector
      ops{{1, compute::SumOp::Make()}, {1, compute::CountOp::Make()}, {1, compute::MeanOp::Make()}};
  CHECK_CYLON_STATUS(mapred::MapredHashGroupBy(table, {0}, ops, &output));

  auto exp_key = ArrayFromJSON(type, "[0, 1, 2, 3, 4]");
  auto exp_sum = ArrayFromJSON(type, "[0, 1, 5, 15, 34]");
  auto exp_cnt = ArrayFromJSON(arrow::int64(), "[1, 1, 2, 3, 4]");
  auto avg = ArrayFromJSON(arrow::float64(), "[0, 1, 2.5, 5, 8.5]");
  auto exp_avg = *arrow::compute::Cast(*avg, type, arrow::compute::CastOptions::Unsafe());
  auto exp_tab = arrow::Table::Make(arrow::schema({
                                                      arrow::field("a", type),
                                                      arrow::field("b_sum", type),
                                                      arrow::field("b_count", arrow::int64()),
                                                      arrow::field("b_mean", type)
                                                  }),
                                    {exp_key, exp_sum, exp_cnt, exp_avg});

  CHECK_ARROW_EQUAL(exp_tab, output->get_table());
}

TEMPLATE_LIST_TEST_CASE("scalar aggregate", "[compute]", ArrowNumericTypes) {
  auto type = default_type_instance<TestType>();
  INFO("testing " + type->ToString());

  auto mult = *arrow::MakeScalar(RANK + 1)->CastTo(type);
  auto base_arr = ArrayFromJSON(type, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]");
  // arr = (rank + 1)*[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  auto arr = arrow::compute::Multiply(base_arr, mult)->make_array();

  auto val = Column::Make(std::move(arr));
  std::shared_ptr<Scalar> res;

  arrow::ArrayVector arrays;
  for (int r = 1; r < WORLD_SZ + 1; r++) {
    mult = *arrow::MakeScalar(r)->CastTo(type);
    arrays.emplace_back(arrow::compute::Multiply(base_arr, mult)->make_array());
  }
  auto global_arr = *arrow::ChunkedArray::Make(arrays, type);

  SECTION("sum") {
    CHECK_CYLON_STATUS(compute::Sum(ctx, val, &res));

//    auto exp = *arrow::MakeScalar((WORLD_SZ * (WORLD_SZ + 1) / 2) * (10 * 11 / 2))->CastTo(type);
    auto exp = *arrow::compute::Sum(global_arr)->scalar()->CastTo(type);
    CHECK_ARROW_EQUAL(exp, res->data());
  }

  SECTION("min") {
    CHECK_CYLON_STATUS(compute::Min(ctx, val, &res));

    auto exp = *arrow::MakeScalar(0)->CastTo(type);
    CHECK_ARROW_EQUAL(exp, res->data());
  }

  SECTION("max") {
    CHECK_CYLON_STATUS(compute::Max(ctx, val, &res));

    auto exp = *arrow::MakeScalar(WORLD_SZ * 10)->CastTo(type);
    CHECK_ARROW_EQUAL(exp, res->data());
  }

  SECTION("count") {
    CHECK_CYLON_STATUS(compute::Count(ctx, val, &res));

    auto exp = std::make_shared<arrow::Int64Scalar>(val->length() * WORLD_SZ);
    CHECK_ARROW_EQUAL(exp, res->data());
  }

  SECTION("mean") {
    CHECK_CYLON_STATUS(compute::Mean(ctx, val, &res));

//    double mean = (WORLD_SZ + 1) * 10. * 11. / (4. * double(val->length()));
    auto exp = arrow::compute::Mean(global_arr)->scalar();
    CHECK_ARROW_EQUAL(exp, res->data());
  }

  SECTION("var") {
    for (int ddof: {0, 1}) {
      SECTION("ddof =" + std::to_string(ddof)) {
        CHECK_CYLON_STATUS(compute::Variance(ctx, val, &res, compute::VarKernelOptions(ddof)));

        auto exp =
            arrow::compute::Variance(global_arr, arrow::compute::VarianceOptions(ddof))->scalar();
        CHECK_ARROW_EQUAL(exp, res->data());
      }
    }
  }

  SECTION("stddev") {
    for (int ddof: {0, 1}) {
      SECTION("ddof =" + std::to_string(ddof)) {
        CHECK_CYLON_STATUS(compute::StdDev(ctx, val, &res, compute::VarKernelOptions(ddof)));

        auto exp =
            arrow::compute::Stddev(global_arr, arrow::compute::VarianceOptions(ddof))->scalar();
        CHECK_ARROW_EQUAL(exp, res->data());
      }
    }
  }
}

}
}
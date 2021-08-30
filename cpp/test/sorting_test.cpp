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
#include <random>

#include <cylon/net/mpi/mpi_communicator.hpp>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/util/builtins.hpp>
#include <cylon/table.hpp>
#include <cylon/util/arrow_utils.hpp>
#include <cylon/groupby/groupby.hpp>
#include <cylon/compute/aggregates.hpp>

#include "common/test_header.hpp"

Status create_table(std::shared_ptr<cylon::Table> &table) {
  arrow::Int64Builder b0;
  for (const auto &x: {0, 1, 4, 3, 2, 3}) {
    b0.Append(x);
  }

  arrow::StringBuilder b1;
  for (const auto &x: {"d", "a", "b", "e", "c", "b"}) {
    b1.Append(x);
  }

  const std::shared_ptr<arrow::Schema>
      &schema = arrow::schema({arrow::field("A", arrow::int64()), arrow::field("B", arrow::binary())});

  auto atable = arrow::Table::Make(schema, {b0.Finish().ValueOrDie(), b1.Finish().ValueOrDie()});

  Table::FromArrowTable(ctx, atable, table);

  return Status::OK();
}

TEST_CASE("sort testing", "[sort]") {
  LOG(INFO) << "Testing sorting";

  std::shared_ptr<cylon::Table> table, output;
  auto status = create_table(table);

  REQUIRE((status.is_ok() && table->Columns() == 2 && table->Rows() == 6));

  std::shared_ptr<cylon::compute::Result> result;

  SECTION("testing ascending") {
    status = cylon::Sort(table, 0, output);
    REQUIRE(status.is_ok());

    const auto &carr = std::static_pointer_cast<arrow::Int64Array>(output->get_table()->column(0)->chunk(0));
    for (int64_t i = 0; i < output->Rows() - 1; i++) {
      REQUIRE((carr->Value(i) <= carr->Value(i + 1)));
    }

    status = cylon::Sort(table, 1, output);
    REQUIRE(status.is_ok());

    const auto &carr1 = std::static_pointer_cast<arrow::StringArray>(output->get_table()->column(1)->chunk(0));
    for (int64_t i = 0; i < output->Rows() - 1; i++) {
      REQUIRE((carr1->GetString(i) <= carr1->GetString(i + 1)));
    }
  }
}



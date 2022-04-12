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

#include "common/test_header.hpp"

#include <arrow/testing/random.h>
#include <cylon/compute/aggregates.hpp>
#include <cylon/util/arrow_rand.hpp>

namespace cylon {
namespace test {

TEST_CASE("table ops testing", "[table_ops]") {
  const int size = 12;
  std::shared_ptr<cylon::Table> input, select;
  CHECK_CYLON_STATUS(cylon::test::CreateTable(ctx, size, input));

  SECTION("table creation") {
    REQUIRE((input->Columns() == 2 && input->Rows() == size));
  }

  SECTION("testing select") {
    CHECK_CYLON_STATUS(Select(input,
                              [](cylon::Row row) { return row.GetInt32(0) % 2 == 0; },
                              select));
    REQUIRE((select->Columns() == 2 && select->Rows() == size / 2));
  }

  SECTION("testing shuffle") {
    CHECK_CYLON_STATUS(Shuffle(input, {0}, select));
    REQUIRE(select->Columns() == 2);
    CheckGlobalSumEqual<int64_t>(ctx, size * WORLD_SZ, select->Rows());
  }

  SECTION("testing unique") {
    std::shared_ptr<cylon::Table> input1, output, sort_table;
    auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

    std::string test_file = "../data/input/duplicate_data_0.csv";
    std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
    CHECK_CYLON_STATUS(cylon::FromCSV(ctx, test_file, input1, read_options));

    std::vector<int> cols = {0};
    CHECK_CYLON_STATUS(cylon::Unique(input1, cols, output, true));

    CHECK_CYLON_STATUS(cylon::Sort(output, 3, sort_table));

    std::shared_ptr<arrow::Table> artb;
    sort_table->ToArrowTable(artb);

    std::vector<int32_t> outval3 = {1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15};
    int count = 0;

    const std::shared_ptr<arrow::Int64Array>
        &carr = std::static_pointer_cast<arrow::Int64Array>(artb->column(3)->chunk(0));
    for (int i = 0; i < carr->length(); i++) {
      INFO(carr->Value(i));
      if (carr->Value(i) == outval3.at(i)) {
        count++;
      }
    }

    REQUIRE((unsigned) count == outval3.size());
  }
}

TEST_CASE("Test unique", "[table_ops]") {
  auto schema = ::arrow::schema({
                                    {field("a", arrow::uint8())},
                                    {field("b", arrow::uint32())},
                                });
  auto table = TableFromJSON(schema, {R"([{"a": null, "b": 5},
                                     {"a": 1,    "b": 3},
                                     {"a": 3,    "b": null}
                                    ])",
                                      R"([{"a": 3, "b": null},
                                     {"a": null,    "b": 5},
                                     {"a": 1,    "b": 3}
                                    ])"});
  std::shared_ptr<Table> in, out;
  CHECK_CYLON_STATUS(Table::FromArrowTable(ctx, table, in));

  SECTION("keep first") {
    auto expected = TableFromJSON(schema, {R"([{"a": null, "b": 5},
                                     {"a": 1,    "b": 3},
                                     {"a": 3,    "b": null}
                                    ])"});

    CHECK_CYLON_STATUS(Unique(in, {0, 1}, out));
    CHECK_ARROW_EQUAL(expected, out->get_table());
  }

  SECTION("keep last") {
    auto expected = TableFromJSON(schema, {R"([{"a": 1,    "b": 3},
                                      {"a": null,    "b": 5},
                                      {"a": 3, "b": null}])"});

    CHECK_CYLON_STATUS(Unique(in, {0, 1}, out, /*first=*/false));
    CHECK_ARROW_EQUAL(expected, out->get_table());
  }
}

}
}

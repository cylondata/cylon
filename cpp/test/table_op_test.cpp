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
#include "test_utils.hpp"

#include <cylon/compute/aggregates.hpp>

using namespace cylon;

TEST_CASE("table ops testing", "[table_ops]") {
cylon::Status status;
const int size = 12;
std::shared_ptr<cylon::Table> input, select;
status = cylon::test::CreateTable(ctx, size, input);

SECTION("table creation") {
  REQUIRE((status.is_ok() && input->Columns()== 2 && input->Rows()== size));
}

SECTION("testing select") {
  status = Select(input, [](cylon::Row row) { return row.GetInt32(0) % 2 == 0;}, select);

  REQUIRE((status.is_ok() && select-> Columns() == 2 && select-> Rows()== size / 2));
}

SECTION("testing shuffle") {

  status = Shuffle(input, {0}, select);
  REQUIRE((status.is_ok() && select->Columns()== 2));

  std::shared_ptr<compute::Result> result;
  status = compute::Count(select, 0, result);
  auto s = std::static_pointer_cast<arrow::Int64Scalar>(result->GetResult().scalar());
  REQUIRE((status.is_ok() && s->value == size *WORLD_SZ));
}

SECTION("testing unique") {

  std::shared_ptr<cylon::Table> input1, output, sort_table;
  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);

  std::string test_file = "/tmp/duplicate_data_0.csv";
  std::cout << "Reading File [" << ctx->GetRank() << "] : " << test_file << std::endl;
  status = cylon::FromCSV(ctx, test_file, input1, read_options);

  REQUIRE(status.is_ok());

  std::vector<int> cols = {0};
  status = cylon::Unique(input1, cols, output, true);

  REQUIRE(status.is_ok());

  status = cylon::Sort(output, 3, sort_table);

  REQUIRE(status.is_ok());

  std::shared_ptr<arrow::Table> artb;
  sort_table->ToArrowTable(artb);

  std::vector<int32_t> outval3 = {1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15};
  int count = 0;

  const std::shared_ptr<arrow::Int64Array>
      &carr = std::static_pointer_cast<arrow::Int64Array>(artb->column(3)->chunk(0));
  for (int i = 0; i < carr->length(); i++) {
    std::cout << carr->Value(i) << std::endl;
    if (carr->Value(i) == outval3.at(i)) {
      count++;
    }
  }

  REQUIRE((unsigned)count == outval3.size());

}

}

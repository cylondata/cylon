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

#include <cylon/table.hpp>

#include "common/test_header.hpp"
#include "test_utils.hpp"

namespace cylon {
void testDistSort(const std::vector<int>& sort_cols,
                  const std::vector<bool>& sort_order,
                  std::shared_ptr<Table>& global_table,
                  std::shared_ptr<Table>& table) {
  std::shared_ptr<Table> out;
  auto ctx = table->GetContext();
  std::shared_ptr<arrow::Table> arrow_output;

  auto status = DistributedSort(table, sort_cols, out, sort_order,
                                {0, 0, SortOptions::INITIAL_SAMPLE});
  REQUIRE(status.is_ok());
  std::shared_ptr<Table> out2;
  bool eq;

  if (RANK == 0) {
    status = Sort(global_table, sort_cols, out2, sort_order);
  } else {
    auto pool = cylon::ToArrowPool(ctx);
    std::shared_ptr<arrow::Table> arrow_empty_table;
    auto arrow_status = util::CreateEmptyTable(
        global_table->get_table()->schema(), &arrow_empty_table, pool);
    out2 = std::make_shared<Table>(ctx, arrow_empty_table);
  }
  std::shared_ptr<Table> out3;
  status = Repartition(out2, &out3);
  status = DistributedEquals(out3, out, eq);
  REQUIRE(eq);
}

namespace test {
TEMPLATE_LIST_TEST_CASE("Dist sort testing", "[dist sort]", ArrowNumericTypes) {
  auto type = default_type_instance<TestType>();
  auto schema = arrow::schema({{arrow::field("a", type)},
                               {arrow::field("b", arrow::float32())}});
  auto global_arrow_table = TableFromJSON(schema, {R"([{"a":  3, "b":0.025},
                                         {"a": 26, "b":0.394},
                                         {"a": 51, "b":0.755},
                                         {"a": 20, "b":0.030},
                                         {"a": 33, "b":0.318},
                                         {"a": 12, "b":0.813},
                                         {"a": 72, "b":0.968},
                                         {"a": 29, "b":0.291},
                                         {"a": 41, "b":0.519},
                                         {"a": 29, "b":0.291},
                                         {"a": 41, "b":0.519},
                                         {"a": 43, "b":0.419},
                                         {"a": 57, "b":0.153},
                                         {"a": 25, "b":0.479},
                                         {"a": 26, "b":0.676},
                                         {"a": 70, "b":0.504},
                                         {"a":  7, "b":0.232},
                                         {"a": 45, "b":0.734},
                                         {"a": 61, "b":0.685},
                                         {"a": 57, "b":0.314},
                                         {"a": 59, "b": 0.837},
                                         {"a": 67, "b": 0.086},
                                         {"a": 14, "b": 0.193},
                                         {"a": 21, "b": 0.853},
                                         {"a": 10, "b": 0.808},
                                         {"a": 13, "b": 0.085},
                                         {"a": 31, "b": 0.122},
                                         {"a": 20, "b": 0.689},
                                         {"a": 37, "b": 0.491},
                                         {"a": 62, "b": 0.262},
                                         {"a": 1 , "b": 0.868},
                                         {"a": 19, "b": 0.422},
                                         {"a": 64, "b": 0.528},
                                         {"a": 37, "b": 0.834},
                                         {"a": 33, "b": 0.010},
                                         {"a": 76, "b": 0.927},
                                         {"a": 4 , "b": 0.529},
                                         {"a": 13, "b": 0.201},
                                         {"a": 45, "b": 0.898},
                                         {"a": 67, "b": 0.407}])"});

  int64_t rows_per_tab = global_arrow_table->num_rows() / WORLD_SZ;
  std::shared_ptr<Table> table1;
  CHECK_CYLON_STATUS(Table::FromArrowTable(
      ctx, global_arrow_table->Slice(RANK * rows_per_tab, rows_per_tab),
      table1));
  std::shared_ptr<Table> global_table;
  CHECK_CYLON_STATUS(
      Table::FromArrowTable(ctx, global_arrow_table, global_table));

  SECTION("dist_sort_test_1") {
    testDistSort({0, 1}, {1, 1}, global_table, table1);
  }

  SECTION("dist_sort_test_2_different_direction") {
    testDistSort({0, 1}, {1, 0}, global_table, table1);
  }

  SECTION("dist_sort_test_3_different_order") {
    testDistSort({1, 0}, {0, 0}, global_table, table1);
  }

  SECTION("dist_sort_test_4_one_empty_table") {
    if (RANK == 0) {
      auto pool = cylon::ToArrowPool(ctx);

      std::shared_ptr<arrow::Table> arrow_empty_table;
      auto arrow_status = util::CreateEmptyTable(table1->get_table()->schema(),
                                                 &arrow_empty_table, pool);
      auto empty_table = std::make_shared<Table>(ctx, arrow_empty_table);
      table1 = empty_table;
    }

    std::shared_ptr<Table> out, out2;
    auto ctx = table1->GetContext();
    std::shared_ptr<arrow::Table> arrow_output;
    auto status = DistributedSort(table1, {1, 0}, out, {0, 0});
    REQUIRE(status.is_ok());
    status = DistributedSort(table1, {1, 0}, out2, {0, 0},
                    {0, 0, SortOptions::INITIAL_SAMPLE});
    REQUIRE(status.is_ok());
    bool eq;
    status = DistributedEquals(out, out2, eq);
    REQUIRE(eq);
  }
}

TEST_CASE("Binary search testing", "[binary search]") {
  auto schema = arrow::schema({{arrow::field("a", arrow::uint32())},
                               {arrow::field("b", arrow::float32())}});

  auto arrow_table = TableFromJSON(schema, {R"([ {"a":  1, "b": 0.868000},
    {"a":  3, "b": 0.025000},
    {"a":  4, "b": 0.529000},
    {"a":  7, "b": 0.232000},
    {"a": 10, "b": 0.808000},
    {"a": 12, "b": 0.813000},
    {"a": 13, "b": 0.085000},
    {"a": 13, "b": 0.201000},
    {"a": 14, "b": 0.193000},
    {"a": 19, "b": 0.422000},
    {"a": 20, "b": 0.030000},
    {"a": 20, "b": 0.689000},
    {"a": 21, "b": 0.853000},
    {"a": 25, "b": 0.479000},
    {"a": 26, "b": 0.394000},
    {"a": 26, "b": 0.676000},
    {"a": 29, "b": 0.291000},
    {"a": 29, "b": 0.291000},
    {"a": 31, "b": 0.122000},
    {"a": 33, "b": 0.010000},
    {"a": 33, "b": 0.318000},
    {"a": 37, "b": 0.491000},
    {"a": 37, "b": 0.834000},
    {"a": 41, "b": 0.519000},
    {"a": 41, "b": 0.519000},
    {"a": 43, "b": 0.419000},
    {"a": 45, "b": 0.734000},
    {"a": 45, "b": 0.898000},
    {"a": 51, "b": 0.755000},
    {"a": 57, "b": 0.153000},
    {"a": 57, "b": 0.314000},
    {"a": 59, "b": 0.837000},
    {"a": 61, "b": 0.685000},
    {"a": 62, "b": 0.262000},
    {"a": 64, "b": 0.528000},
    {"a": 67, "b": 0.086000},
    {"a": 67, "b": 0.407000},
    {"a": 70, "b": 0.504000},
    {"a": 72, "b": 0.968000},
    {"a": 76, "b": 0.927000}])"});

  std::shared_ptr<Table> table;
  CHECK_CYLON_STATUS(Table::FromArrowTable(ctx, arrow_table, table));

  auto arrow_split_points =
      TableFromJSON(schema, {R"([ {"a":  1, "b": 0.068000},
                                  {"a":  4, "b": 0.529000},
                                  {"a":  27, "b": 0.025000},
                                  {"a":  80, "b": 0.232000}])"});

  std::shared_ptr<Table> split_points;
  CHECK_CYLON_STATUS(Table::FromArrowTable(ctx, arrow_split_points, split_points));


  std::unique_ptr<DualTableRowIndexEqualTo> equal_to;
  auto status = DualTableRowIndexEqualTo::Make(
      table->get_table(), split_points->get_table(), {0, 1},
      {0, 1}, &equal_to, {1, 1});

  SECTION("search target less than table's smallest entry") {
    REQUIRE(tableBinarySearch(split_points, table, equal_to, 0) == 0);
  }

  SECTION("table contains search target") {
    REQUIRE(tableBinarySearch(split_points, table, equal_to, 1) == 2);
  }

  
  SECTION("search target not exsit but within range of table") {
    REQUIRE(tableBinarySearch(split_points, table, equal_to, 2) == 16);
  }

  
  SECTION("search target greater than table's largest entry") {
    REQUIRE(tableBinarySearch(split_points, table, equal_to, 3) == table->Rows());
  }
}

TEST_CASE("Merge testing", "[table merge]") {
  auto schema = arrow::schema({{arrow::field("a", arrow::uint32())},
                               {arrow::field("b", arrow::float32())}});

  auto arrow_table_1 = TableFromJSON(schema, {R"([ {"a":  1, "b": 0.868000},
    {"a":  3, "b": 0.025000},
    {"a":  4, "b": 0.529000},
    {"a":  7, "b": 0.232000},
    {"a": 10, "b": 0.808000},
    {"a": 12, "b": 0.813000},
    {"a": 13, "b": 0.085000},
    {"a": 13, "b": 0.201000},
    {"a": 14, "b": 0.193000},
    {"a": 19, "b": 0.422000},
    {"a": 20, "b": 0.030000},
    {"a": 20, "b": 0.689000},
    {"a": 21, "b": 0.853000},
    {"a": 25, "b": 0.479000},
    {"a": 26, "b": 0.394000},
    {"a": 61, "b": 0.685000},
    {"a": 62, "b": 0.262000},
    {"a": 64, "b": 0.528000},
    {"a": 72, "b": 0.968000},
    {"a": 76, "b": 0.927000}])"});

  std::shared_ptr<Table> table1;
  CHECK_CYLON_STATUS(Table::FromArrowTable(ctx, arrow_table_1, table1));

  auto arrow_table_2 = TableFromJSON(schema, {R"([ {"a": 26, "b": 0.676000},
    {"a": 29, "b": 0.291000},
    {"a": 29, "b": 0.291000},
    {"a": 31, "b": 0.122000},
    {"a": 33, "b": 0.010000},
    {"a": 33, "b": 0.318000},
    {"a": 37, "b": 0.491000},
    {"a": 37, "b": 0.834000},
    {"a": 41, "b": 0.519000},
    {"a": 41, "b": 0.519000},
    {"a": 43, "b": 0.419000},
    {"a": 45, "b": 0.734000},
    {"a": 45, "b": 0.898000},
    {"a": 51, "b": 0.755000},
    {"a": 57, "b": 0.153000},
    {"a": 57, "b": 0.314000},
    {"a": 59, "b": 0.837000},
    {"a": 67, "b": 0.086000},
    {"a": 67, "b": 0.407000},
    {"a": 70, "b": 0.504000}])"});

  std::shared_ptr<Table> table2;
  CHECK_CYLON_STATUS(Table::FromArrowTable(ctx, arrow_table_2, table2));

  std::shared_ptr<Table> concat, sorted, merged;
  Merge({table1, table2}, concat);
  Sort(concat, {0, 1}, sorted, {1, 1});

  MergeSortedTable({table1, table2}, {0, 1}, {1, 1}, merged, true);

  bool result;
  Equals(sorted, merged, result);
  REQUIRE(result);
}

}  // namespace test
}  // namespace cylon
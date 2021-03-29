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

#include <partition/partition.hpp>
#include <util/murmur3.hpp>
#include <compute/aggregates.hpp>

using namespace cylon;

TEST_CASE("Partition testing", "[join]") {
  const int rows = 12;

  cylon::Status status;
  std::shared_ptr<cylon::Table> table;
  std::shared_ptr<cylon::Table> output;
  status = cylon::test::CreateTable(ctx, rows, table);

  SECTION("modulo partition test") {
    std::vector<uint32_t> partitions, count;
    status = cylon::MapToHashPartitions(table, 0, WORLD_SZ, partitions, count);
    REQUIRE((status.is_ok() && (partitions.size() == rows) && (count.size() == (size_t) WORLD_SZ)));

    const std::shared_ptr<arrow::Int32Array> &arr
        = std::static_pointer_cast<arrow::Int32Array>(table->get_table()->column(0)->chunk(0));
    for (int i = 0; i < table->Rows(); i++) {
      REQUIRE(partitions[i] == (uint32_t) (arr->Value(i) % WORLD_SZ));
    }

    uint32_t sum = std::accumulate(count.begin(), count.end(), 0);
    REQUIRE(sum == table->Rows());

    std::vector<std::shared_ptr<arrow::Table>> split_tables;
    status = cylon::Split(table, WORLD_SZ, partitions, count, split_tables);
    REQUIRE((status.is_ok() && (split_tables.size() == (size_t) WORLD_SZ)));

    for (int i = 0; i < WORLD_SZ; i++) {
      REQUIRE(split_tables[i]->num_rows() == count[i]);
    }
  }

  SECTION("hash partition test") {
    std::vector<uint32_t> partitions, count;
    status = cylon::MapToHashPartitions(table, 1, WORLD_SZ, partitions, count);
    REQUIRE((status.is_ok() && (partitions.size() == rows) && (count.size() == (size_t) WORLD_SZ)));

    const std::shared_ptr<arrow::DoubleArray> &arr
        = std::static_pointer_cast<arrow::DoubleArray>(table->get_table()->column(1)->chunk(0));
    for (int i = 0; i < table->Rows(); i++) {
      double v = arr->Value(i);
      uint32_t hash = 0;
      cylon::util::MurmurHash3_x86_32(&v, sizeof(double), 0, &hash);
      REQUIRE(partitions[i] == (uint32_t) (hash % WORLD_SZ));
    }

    uint32_t sum = std::accumulate(count.begin(), count.end(), 0);
    REQUIRE(sum == table->Rows());

    std::vector<std::shared_ptr<arrow::Table>> split_tables;
    status = cylon::Split(table, WORLD_SZ, partitions, count, split_tables);
    REQUIRE((status.is_ok() && (split_tables.size() == (size_t) WORLD_SZ)));

    for (int i = 0; i < WORLD_SZ; i++) {
      REQUIRE(split_tables[i]->num_rows() == count[i]);
    }
  }

  SECTION("range partition test") {
    std::vector<uint32_t> partitions, count;
    status = cylon::MapToSortPartitions(table, 0, WORLD_SZ, partitions, count, true, table->Rows(), WORLD_SZ);
    LOG(INFO) << "stat " << status.get_code() << " " << status.get_msg() ;
    REQUIRE((status.is_ok() && (partitions.size() == rows) && (count.size() == (size_t) WORLD_SZ)));

    for (int i = 0; i < table->Rows() - 1; i++) {
      REQUIRE(partitions[i] <= partitions[i + 1]);
    }

    uint32_t sum = std::accumulate(count.begin(), count.end(), 0);
    REQUIRE(sum == table->Rows());

    std::vector<std::shared_ptr<arrow::Table>> split_tables;
    status = cylon::Split(table, WORLD_SZ, partitions, count, split_tables);
    REQUIRE((status.is_ok() && (split_tables.size() == (size_t) WORLD_SZ)));

    for (int i = 0; i < WORLD_SZ; i++) {
      REQUIRE(split_tables[i]->num_rows() == count[i]);
    }
  }

  SECTION("dist sort test") {
    status = cylon::DistributedSort(table, 0, output, true, {(uint32_t) WORLD_SZ, (uint64_t) table->Rows()});
    REQUIRE((status.is_ok()));

    for (auto &arr: output->get_table()->column(0)->chunks()) {
      const std::shared_ptr<arrow::Int32Array> &carr = std::static_pointer_cast<arrow::Int32Array>(arr);
      for (int i = 0; i < carr->length() - 1; i++) {
        REQUIRE((carr->Value(i) <= carr->Value(i + 1)));
      }
    }

    std::shared_ptr<cylon::compute::Result> res;
    status = cylon::compute::Count(output, 0, res);
    auto scalar = std::static_pointer_cast<arrow::Int64Scalar>(res->GetResult().scalar());
    REQUIRE((status.is_ok() && scalar->value == table->Rows() * WORLD_SZ));

    output.reset();

    status = cylon::DistributedSort(table, 1, output);
    REQUIRE((status.is_ok()));

    for (auto &arr: output->get_table()->column(1)->chunks()) {
      const std::shared_ptr<arrow::DoubleArray> &carr = std::static_pointer_cast<arrow::DoubleArray>(arr);
      for (int i = 0; i < carr->length() - 1; i++) {
        REQUIRE((carr->Value(i) <= carr->Value(i + 1)));
      }
    }

    status = cylon::compute::Count(output, 1, res);
    scalar = std::static_pointer_cast<arrow::Int64Scalar>(res->GetResult().scalar());
    REQUIRE((status.is_ok() && scalar->value == table->Rows() * WORLD_SZ));

    output.reset();

    status = cylon::DistributedSort(table, {0, 1}, output, {true, false},
                                    {(uint32_t) WORLD_SZ, (uint64_t) table->Rows()});
    REQUIRE((status.is_ok()));

    for (int c = 0; c < output->get_table()->column(0)->num_chunks(); c++) {
      const auto &carr1 = std::static_pointer_cast<arrow::Int32Array>(output->get_table()->column(0)->chunk(c));
      const auto &carr2 = std::static_pointer_cast<arrow::DoubleArray>(output->get_table()->column(1)->chunk(c));
      for (int i = 0; i < carr1->length() - 1; i++) {
        REQUIRE((carr1->Value(i) <= carr1->Value(i + 1)));
        if (carr1->Value(i) == carr1->Value(i + 1)) REQUIRE((carr2->Value(i) >= carr2->Value(i + 1)));
      }
    }

    status = cylon::compute::Count(output, 0, res);
    scalar = std::static_pointer_cast<arrow::Int64Scalar>(res->GetResult().scalar());
    REQUIRE((status.is_ok() && scalar->value == table->Rows() * WORLD_SZ));
  }
}

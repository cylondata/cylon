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

#include "cylon/serialize/table_serialize.hpp"
#include "cylon/arrow/arrow_buffer.hpp"

namespace cylon {
namespace test {

TEST_CASE("serialize table", "[serialization]") {
  auto schema = arrow::schema({
                                  {field("_", arrow::boolean())},
                                  {field("a", arrow::uint32())},
                                  {field("b", arrow::float64())},
                                  {field("c", arrow::utf8())},
                              });

  auto in_table = TableFromJSON(schema, {R"([{"_": true,  "a": null, "b": 5,  "c": "1"},
                                     {"_": false,  "a": 1,    "b": 3,    "c": "12"},
                                     {"_": true,  "a": 3,    "b": null, "c": "123"},
                                     {"_": null,  "a": null, "b": null, "c": null},
                                     {"_": true,  "a": 2,    "b": 5,    "c": "1234"},
                                     {"_": false,  "a": 1,    "b": 5,    "c": null}
                                    ])"});

  for (const auto &atable: {in_table, in_table->Slice(3)}) {
    SECTION(atable.get() == in_table.get() ? "without offset" : "with offset") {
      auto table = std::make_shared<Table>(ctx, atable);
      std::shared_ptr<TableSerializer> ser;
      CHECK_CYLON_STATUS(CylonTableSerializer::Make(table, &ser));

      // table serializer only has pointers to data. To emulate gather/ all gather behavior, create a
      // vector of arrow buffers.
      std::vector<std::shared_ptr<Buffer>> buffers;
      buffers.reserve(ser->getNumberOfBuffers());
      const auto &data_buffers = ser->getDataBuffers();
      const auto &buffer_sizes = ser->getBufferSizes();
      for (int i = 0; i < ser->getNumberOfBuffers(); i++) {
        arrow::BufferBuilder builder;
        REQUIRE(builder.Append(data_buffers[i], buffer_sizes[i]).ok());
        buffers.push_back(std::make_shared<ArrowBuffer>(builder.Finish().ValueOrDie()));
      }

      std::shared_ptr<Table> output;
      CHECK_CYLON_STATUS(DeserializeTable(ctx, schema, buffers, buffer_sizes, &output));

      CHECK_ARROW_EQUAL(atable, output->get_table());
    }
  }
}

TEST_CASE("serialize column", "[serialization]") {
  auto schema = arrow::schema({
                                  {field("_", arrow::boolean())},
                                  {field("a", arrow::uint32())},
                                  {field("b", arrow::float64())},
                                  {field("c", arrow::utf8())},
                              });

  auto in_table = TableFromJSON(schema, {R"([{"_": true,  "a": null, "b": 5,  "c": "1"},
                                     {"_": false,  "a": 1,    "b": 3,    "c": "12"},
                                     {"_": true,  "a": 3,    "b": null, "c": "123"},
                                     {"_": null,  "a": null, "b": null, "c": null},
                                     {"_": true,  "a": 2,    "b": 5,    "c": "1234"},
                                     {"_": false,  "a": 1,    "b": 5,    "c": null}
                                    ])"});

  for (const auto &atable: {in_table, in_table->Slice(3)}) {
    SECTION(atable.get() == in_table.get() ? "without offset" : "with offset") {

      for (const auto &col: atable->columns()) {
        if (col->type()->id() == arrow::Type::BOOL){
          continue;
        }

        std::shared_ptr<ColumnSerializer> ser;
        CHECK_CYLON_STATUS(CylonColumnSerializer::Make(col, &ser));

        // table serializer only has pointers to data. To emulate gather/ all gather behavior, create a
        // vector of arrow buffers.
        std::array<std::shared_ptr<Buffer>, 3> buffers{};
        const auto &data_buffers = ser->getDataBuffers();
        const auto &buffer_sizes = ser->getBufferSizes();
        for (size_t i = 0; i < 3; i++) {
          arrow::BufferBuilder builder;
          REQUIRE(builder.Append(data_buffers[i], buffer_sizes[i]).ok());
          buffers[i] = std::make_shared<ArrowBuffer>(builder.Finish().ValueOrDie());
        }

        std::shared_ptr<Column> output;
        CHECK_CYLON_STATUS(DeserializeColumn(col->type(), buffers, buffer_sizes, &output));

        CHECK_ARROW_EQUAL(col->chunk(0), output->data());
      }
    }
  }
}

}
}
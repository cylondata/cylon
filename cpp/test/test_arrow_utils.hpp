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

#pragma once

#include <arrow/api.h>
#include <arrow/ipc/api.h>

#define ARROW_ABORT_NOT_OK(expr)                                    \
  do {                                                              \
    auto _res = (expr);                                             \
    ::arrow::Status _st = ::arrow::internal::GenericToStatus(_res); \
    if (ARROW_PREDICT_FALSE(!_st.ok())) {                           \
      _st.Abort();                                                  \
    }                                                               \
  } while (false);

namespace cylon {
namespace test {

/**
 * Testing type tuples for templated tests
 */
using ArrowNumericTypes = std::tuple<arrow::Int8Type, arrow::Int16Type, arrow::Int32Type, arrow::Int64Type,
                                     arrow::UInt8Type, arrow::UInt16Type, arrow::UInt32Type, arrow::UInt64Type,
                                     arrow::FloatType, arrow::DoubleType>;

using ArrowTemporalTypes = std::tuple<arrow::Date32Type,
                                      arrow::Date64Type,
                                      arrow::TimestampType,
                                      arrow::Time32Type,
                                      arrow::Time64Type>;

using ArrowBinaryTypes = std::tuple<arrow::StringType, arrow::LargeStringType,
                                    arrow::BinaryType, arrow::LargeBinaryType>;


/**
 * Arrow data structures from strings
 * from: https://github.com/apache/arrow/blob/master/cpp/src/arrow/testing/gtest_util.cc#L406-L456
*/

/*
 Create array
 auto type = ...;
 auto array = ArrayFromJSON(type, R"(["a", "b", "c"])");
 */
std::shared_ptr<arrow::Array> ArrayFromJSON(const std::shared_ptr<arrow::DataType> &type,
                                            arrow::util::string_view json) {
  std::shared_ptr<arrow::Array> out;
  ARROW_ABORT_NOT_OK(arrow::ipc::internal::json::ArrayFromJSON(type, json, &out));
  return out;
}

std::shared_ptr<arrow::Array> DictArrayFromJSON(const std::shared_ptr<arrow::DataType> &type,
                                                arrow::util::string_view indices_json,
                                                arrow::util::string_view dictionary_json) {
  std::shared_ptr<arrow::Array> out;
  ARROW_ABORT_NOT_OK(arrow::ipc::internal::json::DictArrayFromJSON(type, indices_json, dictionary_json, &out));
  return out;
}

/*
 Create chunked array
  auto type = ...;
  auto chunked_array = ChunkedArrayFromJSON(type, {
                                                      "[0, 1]",
                                                      "[3, 2, 1]",
                                                      "[5, 0]",
                                                  });
 */
std::shared_ptr<arrow::ChunkedArray> ChunkedArrayFromJSON(const std::shared_ptr<arrow::DataType> &type,
                                                          const std::vector<std::string> &json) {
  arrow::ArrayVector out_chunks;
  for (const std::string &chunk_json: json) {
    out_chunks.push_back(ArrayFromJSON(type, chunk_json));
  }
  return std::make_shared<arrow::ChunkedArray>(std::move(out_chunks), type);
}

/*
  Create record batch
  auto schema = ::arrow::schema({
      {field("a", decimal128(3, 1))},
      {field("b", decimal256(4, 2))},
  });
  auto batch = RecordBatchFromJSON(schema,
                                 R"([{"a": "12.3", "b": "12.34"},
                                     {"a": "45.6", "b": "12.34"},
                                     {"a": "12.3", "b": "-12.34"},
                                     {"a": "-12.3", "b": null},
                                     {"a": "-12.3", "b": "-45.67"}
                                     ])");
*/
std::shared_ptr<arrow::RecordBatch> RecordBatchFromJSON(const std::shared_ptr<arrow::Schema> &schema,
                                                        arrow::util::string_view json) {
  // Parse as a StructArray
  auto struct_type = struct_(schema->fields());
  std::shared_ptr<arrow::Array> struct_array = ArrayFromJSON(struct_type, json);

  // Convert StructArray to RecordBatch
  return *arrow::RecordBatch::FromStructArray(struct_array);
}

/*
 Create table
  auto schema = ::arrow::schema({
      {field("a", uint8())},
      {field("b", uint32())},
  });
  table = TableFromJSON(schema, {R"([{"a": null, "b": 5},
                                     {"a": 1,    "b": 3},
                                     {"a": 3,    "b": null}
                                    ])",
                                 R"([{"a": null, "b": null},
                                     {"a": 2,    "b": 5},
                                     {"a": 1,    "b": 5}
                                    ])"});
 */
std::shared_ptr<arrow::Table> TableFromJSON(const std::shared_ptr<arrow::Schema> &schema,
                                            const std::vector<std::string> &json) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  for (const std::string &batch_json: json) {
    batches.push_back(RecordBatchFromJSON(schema, batch_json));
  }
  return *arrow::Table::FromRecordBatches(schema, batches);
}


/**
 * from:
 * https://github.com/apache/arrow/blob/master/cpp/src/arrow/compute/kernels/test_util.h#L123-L143
 * Helper to get a default instance of a type, including parameterized types
 */
template<typename T>
arrow::enable_if_parameter_free<T, std::shared_ptr<arrow::DataType>> default_type_instance() {
  return arrow::TypeTraits<T>::type_singleton();
}
template<typename T>
arrow::enable_if_time<T, std::shared_ptr<arrow::DataType>> default_type_instance() {
  // Time32 requires second/milli, Time64 requires nano/micro
  if (bit_width(T::type_id) == 32) {
    return std::make_shared<T>(arrow::TimeUnit::type::SECOND);
  }
  return std::make_shared<T>(arrow::TimeUnit::type::NANO);
}

template<typename T>
arrow::enable_if_timestamp<T, std::shared_ptr<arrow::DataType>> default_type_instance() {
  return std::make_shared<T>(arrow::TimeUnit::type::SECOND);
}

template<typename T>
arrow::enable_if_decimal<T, std::shared_ptr<arrow::DataType>> default_type_instance() {
  return std::make_shared<T>(5, 2);
}

/**
 * Make binary scalar
 * @tparam T
 * @param val
 * @return
 */
template<typename T>
arrow::enable_if_base_binary<T, std::shared_ptr<arrow::Scalar>> MakeBinaryScalar(std::string val) {
  using ScalarT = typename arrow::TypeTraits<T>::ScalarType;
  return std::make_shared<ScalarT>(arrow::Buffer::FromString(std::move(val)));
}

} // namespace test
} // namespace cylon


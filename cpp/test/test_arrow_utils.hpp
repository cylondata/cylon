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

std::shared_ptr<arrow::ChunkedArray> ChunkedArrayFromJSON(const std::shared_ptr<arrow::DataType> &type,
                                                          const std::vector<std::string> &json) {
  arrow::ArrayVector out_chunks;
  for (const std::string &chunk_json: json) {
    out_chunks.push_back(ArrayFromJSON(type, chunk_json));
  }
  return std::make_shared<arrow::ChunkedArray>(std::move(out_chunks), type);
}

std::shared_ptr<arrow::RecordBatch> RecordBatchFromJSON(const std::shared_ptr<arrow::Schema> &schema,
                                                        arrow::util::string_view json) {
  // Parse as a StructArray
  auto struct_type = struct_(schema->fields());
  std::shared_ptr<arrow::Array> struct_array = ArrayFromJSON(struct_type, json);

  // Convert StructArray to RecordBatch
  return *arrow::RecordBatch::FromStructArray(struct_array);
}

std::shared_ptr<arrow::Table> TableFromJSON(const std::shared_ptr<arrow::Schema> &schema,
                                            const std::vector<std::string> &json) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  for (const std::string &batch_json: json) {
    batches.push_back(RecordBatchFromJSON(schema, batch_json));
  }
  return *arrow::Table::FromRecordBatches(schema, batches);
}

} // namespace test
} // namespace cylon


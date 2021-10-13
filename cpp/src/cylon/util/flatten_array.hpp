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
#ifndef CYLON_CPP_SRC_CYLON_UTIL_FLATTEN_ARRAY_HPP_
#define CYLON_CPP_SRC_CYLON_UTIL_FLATTEN_ARRAY_HPP_

#include "cylon/table.hpp"

namespace cylon {

struct ColumnFlattenKernel {
  ColumnFlattenKernel() = default;
  virtual ~ColumnFlattenKernel() = default;
  virtual int32_t ByteWidth() const = 0;

  virtual Status CopyData(uint8_t col_idx,
                          int32_t *row_offset,
                          uint8_t *data_buf,
                          const int32_t *offset_buff) const = 0;

  virtual Status IncrementRowOffset(int32_t *offsets) const = 0;
};

struct FlattenedArray {
  FlattenedArray(std::shared_ptr<arrow::Array> flattened,
                 std::vector<std::shared_ptr<arrow::Array>> parent_data)
      : flattened(std::move(flattened)), parent_data(std::move(parent_data)) {};

  std::shared_ptr<arrow::Array> flattened;
  std::vector<std::shared_ptr<arrow::Array>> parent_data;
};

Status FlattenArrays(CylonContext *ctx,
                     const std::vector<std::shared_ptr<arrow::Array>> &arrays,
                     std::shared_ptr<FlattenedArray> *output);
// count nulls
// create offset array
// copy data to the flattened array
}

#endif //CYLON_CPP_SRC_CYLON_UTIL_FLATTEN_ARRAY_HPP_

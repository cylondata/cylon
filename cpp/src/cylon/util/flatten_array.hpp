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

struct ArraysMetadata {
  uint8_t arrays_with_nulls = 0;
  int32_t fixed_size_bytes_per_row = 0;
  std::vector<uint8_t> var_bin_array_indices{};

  inline bool ContainsNullArrays() const { return arrays_with_nulls > 0; };
  inline bool ContainsOnlyNumeric() const { return var_bin_array_indices.empty(); }
};

struct FlattenedArray {
  FlattenedArray(std::shared_ptr<arrow::Array> flattened,
                 std::vector<std::shared_ptr<arrow::Array>> parent_data, ArraysMetadata metadata)
      : data(std::move(flattened)),
        parent_data(std::move(parent_data)), metadata(std::move(metadata)) {};

  const std::shared_ptr<arrow::Array> data;
  const std::vector<std::shared_ptr<arrow::Array>> parent_data;
  const ArraysMetadata metadata;
};

/**
 * Row-wise flattens a set of arrays to a single Binary array.
 * ex: a1 = [a, b, c, d], a2 = [e, f, g, h] --> [ae, bf, cg, dh]
 *
 * If there are nulls in either of arrays, each element of the output array would be arranged as
 * follows.
 * |  1byte  | each index 1 byte         |    row size in bytes^^   |
 * |num_nulls|<...sparse null indices...>|<...flattened row data...>|
 *
 * ^^ While flattening null values, for fixed sized data (int, float, etc) an empty element (i.e. 0)
 * will be appended. For variable sized data (str, binary), null values would have empty slots.
 *
 * @param ctx
 * @param arrays
 * @param output
 * @return
 */
Status FlattenArrays(CylonContext *ctx,
                     const std::vector<std::shared_ptr<arrow::Array>> &arrays,
                     std::shared_ptr<FlattenedArray> *output);
}

#endif //CYLON_CPP_SRC_CYLON_UTIL_FLATTEN_ARRAY_HPP_
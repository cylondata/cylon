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

#ifndef CYLON_SRC_JOIN_JOIN_UTILS_HPP_
#define CYLON_SRC_JOIN_JOIN_UTILS_HPP_

#include <arrow/api.h>
#include <map>

#include "cylon/status.hpp"

namespace cylon {
namespace join {
namespace util {

std::shared_ptr<arrow::Schema> build_final_table_schema(const std::shared_ptr<arrow::Table> &left_tab,
                                       const std::shared_ptr<arrow::Table> &right_tab,
                                       const std::string &left_table_prefix,
                                       const std::string &right_table_prefix);

Status build_final_table(const std::vector<int64_t> &left_indices,
                                const std::vector<int64_t> &right_indices,
                                const std::shared_ptr<arrow::Table> &left_tab,
                                const std::shared_ptr<arrow::Table> &right_tab,
                                const std::string &left_table_prefix,
                                const std::string &right_table_prefix,
                                std::shared_ptr<arrow::Table> *final_table,
                                arrow::MemoryPool *memory_pool);

Status build_final_table_inplace_index(
    size_t left_inplace_column, size_t right_inplace_column,
    const std::vector<int64_t> &left_indices,
    const std::vector<int64_t> &right_indices,
    std::shared_ptr<arrow::UInt64Array> &left_index_sorted_column,
    std::shared_ptr<arrow::UInt64Array> &right_index_sorted_column,
    const std::shared_ptr<arrow::Table> &left_tab,
    const std::shared_ptr<arrow::Table> &right_tab,
    const std::string &left_table_prefix,
    const std::string &right_table_prefix,
    std::shared_ptr<arrow::Table> *final_table,
    arrow::MemoryPool *memory_pool);

Status CombineChunks(const std::shared_ptr<arrow::Table> &table,
                            int col_index,
                            std::shared_ptr<arrow::Table> &output_table,
                            arrow::MemoryPool *memory_pool);

inline bool is_inplace_join_possible(arrow::Type::type kType) {
  return kType == arrow::Type::UINT8 || kType == arrow::Type::INT8 ||
      kType == arrow::Type::UINT16 || kType == arrow::Type::INT16 ||
      kType == arrow::Type::UINT32 || kType == arrow::Type::INT32 ||
      kType == arrow::Type::UINT64 || kType == arrow::Type::INT64 ||
      kType == arrow::Type::FLOAT || kType == arrow::Type::DOUBLE;
}

}
}
}
#endif //CYLON_SRC_JOIN_JOIN_UTILS_HPP_

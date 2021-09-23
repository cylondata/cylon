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

#ifndef CYLON_CPP_SRC_CYLON_INDEXING_INDEX_UTILS_HPP_
#define CYLON_CPP_SRC_CYLON_INDEXING_INDEX_UTILS_HPP_

#include "cylon/table.hpp"

namespace cylon {
namespace indexing {

/**
 * Slice a table row range [start_index, end_index] and returns a new table with a sliced index
 * @param input_table
 * @param start
 * @param end_inclusive
 * @param columns
 * @param output
 * @return
 */
Status SliceTableByRange(const std::shared_ptr<Table> &input_table,
                         int64_t start,
                         int64_t end_inclusive,
                         std::vector<int> columns,
                         std::shared_ptr<Table> *output,
                         bool reset_index = false);

/**
 * Filter table based on indices. The new table will inherit the indices array as it's index (LinearIndex)
 * @param input_table
 * @param indices
 * @param columns
 * @param output
 * @param bounds_check
 * @return
 */
Status SelectTableByRows(const std::shared_ptr<Table> &input_table,
                         const std::shared_ptr<arrow::Array> &indices,
                         std::vector<int> columns,
                         std::shared_ptr<Table> *output,
                         bool bounds_check = false,
                         bool reset_index = false);
/**
 * Filter table by boolean mask
 * @param input_table
 * @param mask boolean array mask
 * @param columns
 * @param output
 * @param bounds_check
 * @param reset_index
 * @return
 */
Status FilterTableByMask(const std::shared_ptr<Table> &input_table,
                         const std::shared_ptr<arrow::Array> &mask,
                         std::vector<int> columns,
                         std::shared_ptr<Table> *output,
                         bool reset_index = false);

/**
 * Returns a table masked by another table. masked elements will be marked null.
 * (iterate through all the columns of the input table, and set output arrays validity buffer with Logical-And'ed mask)
 * @param input_table
 * @param mask table with boolean arrays
 * @param output
 * @return
 */
Status MaskTable(const std::shared_ptr<Table> &input_table,
                 const std::shared_ptr<Table> &mask,
                 std::shared_ptr<Table> *output);

}
}

#endif //CYLON_CPP_SRC_CYLON_INDEXING_INDEX_UTILS_HPP_

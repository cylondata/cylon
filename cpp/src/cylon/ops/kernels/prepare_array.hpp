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

#ifndef CYLON_SRC_CYLON_OPS_KERNELS_UTILS_PREPAREARRAY_HPP_
#define CYLON_SRC_CYLON_OPS_KERNELS_UTILS_PREPAREARRAY_HPP_
#include <status.hpp>
#include <ctx/cylon_context.hpp>
#include <arrow/table.h>
#include <util/arrow_utils.hpp>
#include <ctx/arrow_memory_pool_utils.hpp>

namespace cylon {
namespace kernel {
/**
 * creates an Arrow array based on col_idx, filtered by row_indices
 * @param ctx
 * @param table
 * @param col_idx
 * @param row_indices
 * @param array_vector
 * @return
 */
Status PrepareArray(std::shared_ptr<CylonContext> &ctx,
                    const std::shared_ptr<arrow::Table> &table,
                    int32_t col_idx,
                    const std::vector<int64_t> &row_indices,
                    arrow::ArrayVector &array_vector);
}
}
#endif //CYLON_SRC_CYLON_OPS_KERNELS_UTILS_PREPAREARRAY_HPP_

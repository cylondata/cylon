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

#include <glog/logging.h>

#include <cylon/util/arrow_utils.hpp>
#include <cylon/ops/kernels/prepare_array.hpp>
#include <cylon/ctx/arrow_memory_pool_utils.hpp>

namespace cylon {
namespace kernel {
Status PrepareArray(std::shared_ptr<CylonContext> &ctx,
                    const std::shared_ptr<arrow::Table> &table,
                    int32_t col_idx,
                    const std::vector<int64_t> &row_indices,
                    arrow::ArrayVector &array_vector) {
  std::shared_ptr<arrow::Array> destination_col_array;
  arrow::Status ar_status = cylon::util::copy_array_by_indices(row_indices,
                                                               cylon::util::GetChunkOrEmptyArray(table->column(col_idx), 0),
                                                               &destination_col_array, cylon::ToArrowPool(ctx));
  if (ar_status != arrow::Status::OK()) {
    LOG(FATAL) << "Failed while copying a column to the final table from tables."
               << ar_status.ToString();
    return Status(static_cast<int>(ar_status.code()), ar_status.message());
  }
  array_vector.push_back(destination_col_array);
  return Status::OK();
}
}
}
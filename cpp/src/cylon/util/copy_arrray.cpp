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

#include <arrow/api.h>
#include <arrow/compute/api.h>

#include <cylon/util/arrow_utils.hpp>

namespace cylon {
namespace util {

arrow::Status copy_array_by_indices(const std::vector<int64_t> &indices,
                                    const std::shared_ptr<arrow::Array> &data_array,
                                    std::shared_ptr<arrow::Array> *copied_array,
                                    arrow::MemoryPool *memory_pool) {
  auto idx_array = util::WrapNumericVector(indices);
  arrow::compute::ExecContext exec_ctx(memory_pool);
  ARROW_ASSIGN_OR_RAISE(*copied_array, arrow::compute::Take(*data_array, *idx_array,
                                                            arrow::compute::TakeOptions::NoBoundsCheck(),
                                                            &exec_ctx));
  return arrow::Status::OK();
}

}  // namespace util
}  // namespace cylon

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

#include "cylon/column.hpp"
#include "cylon/arrow/arrow_types.hpp"
#include "cylon/util/macros.hpp"
#include "cylon/ctx/arrow_memory_pool_utils.hpp"

namespace cylon {

std::shared_ptr<Column> Column::Make(std::shared_ptr<arrow::Array> data_) {
  return std::make_shared<Column>(std::move(data_));
}

Status Column::Make(const std::shared_ptr<CylonContext> &ctx,
                    const std::shared_ptr<arrow::ChunkedArray> &data_,
                    std::shared_ptr<Column> *output) {
  CYLON_ASSIGN_OR_RAISE(auto arr, arrow::Concatenate(data_->chunks(), ToArrowPool(ctx)))
  *output = Column::Make(std::move(arr));
  return Status::OK();
}

Column::Column(std::shared_ptr<arrow::Array> data)
    : type_(tarrow::ToCylonType(data->type())), data_(std::move(data)) {}

const std::shared_ptr<arrow::Array> &Column::data() const { return data_; }

const std::shared_ptr<DataType> &Column::type() const { return type_; }

int64_t Column::length() const { return data_->length(); }

}  // namespace cylon

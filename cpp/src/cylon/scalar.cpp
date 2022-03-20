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

#include "scalar.hpp"
#include "cylon/arrow/arrow_types.hpp"

namespace cylon {

Scalar::Scalar(std::shared_ptr<arrow::Scalar> data)
    : type_(tarrow::ToCylonType(data->type)), data_(std::move(data)) {}

const std::shared_ptr<DataType> &Scalar::type() const { return type_; }

const std::shared_ptr<arrow::Scalar> &Scalar::data() const { return data_; }

std::shared_ptr<Scalar> Scalar::Make(std::shared_ptr<arrow::Scalar> data) {
  return std::make_shared<Scalar>(std::move(data));
}
}
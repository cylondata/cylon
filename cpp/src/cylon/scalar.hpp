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

#ifndef CYLON_CPP_SRC_CYLON_SCALAR_HPP_
#define CYLON_CPP_SRC_CYLON_SCALAR_HPP_

#include <arrow/api.h>

#include "cylon/data_types.hpp"

namespace cylon {

class Scalar {
 public:
  explicit Scalar(std::shared_ptr<arrow::Scalar> data);

  const std::shared_ptr<DataType> &type() const;

  const std::shared_ptr<arrow::Scalar> &data() const;

 private:
  std::shared_ptr<DataType> type_; // The datatype of the column
  std::shared_ptr<arrow::Scalar> data_;
};

}

#endif //CYLON_CPP_SRC_CYLON_SCALAR_HPP_

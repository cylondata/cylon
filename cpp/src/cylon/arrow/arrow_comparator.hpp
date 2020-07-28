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

#ifndef CYLON_SRC_CYLON_ARROW_ARROW_COMPARATOR_HPP_
#define CYLON_SRC_CYLON_ARROW_ARROW_COMPARATOR_HPP_

#include <arrow/api.h>
#include "../status.hpp"

namespace cylon {
class ArrowComparator {
 public:
  virtual int compare(std::shared_ptr<arrow::Array> array1,
                      int64_t index1,
                      std::shared_ptr<arrow::Array> array2,
                      int64_t index2) = 0;
};

std::shared_ptr<ArrowComparator> GetComparator(const std::shared_ptr<arrow::DataType> &type);

/**
 * This Util class can be used to compare the equality  of rows of two tables. This class
 * expect Tables to have just a single chunk in all its columns
 *
 * todo resolve single chunk expectation
 */
class TableRowComparator {
 private:
  std::vector<std::shared_ptr<ArrowComparator>> comparators;
 public:
  explicit TableRowComparator(const std::vector<std::shared_ptr<arrow::Field>> &vector);
  int compare(const std::shared_ptr<arrow::Table> &table1,
              int64_t index1,
              const std::shared_ptr<arrow::Table> &table2,
              int64_t index2);
};

}  // namespace cylon

#endif //CYLON_SRC_CYLON_ARROW_ARROW_COMPARATOR_HPP_

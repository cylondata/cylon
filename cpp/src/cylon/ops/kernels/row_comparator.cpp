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

#include "row_comparator.hpp"

bool cylon::kernel::RowComparator::operator()(const pair<int32_t, int64_t> &record1,
                                               const pair<int32_t, int64_t> &record2) const {
  bool x = this->comparator->compare(this->tables->at(record1.first), record1.second,
                                     this->tables->at(record2.first), record2.second) == 0;
  return x;
}

size_t cylon::kernel::RowComparator::operator()(const pair<int32_t, int64_t> &record) const {
  size_t hash = this->row_hashing_kernel->Hash(this->tables->at(record.first), record.second);
  return hash;
}

cylon::kernel::RowComparator::RowComparator(std::shared_ptr<CylonContext> ctx,
                                              std::shared_ptr<std::vector<std::shared_ptr<arrow::Table>>> tables,
                                              std::shared_ptr<arrow::Schema> schema) {
  this->tables = tables;
  this->comparator = std::make_shared<cylon::TableRowComparator>(schema->fields());
  this->row_hashing_kernel = std::make_shared<cylon::RowHashingKernel>(schema->fields(), cylon::ToArrowPool(ctx.get()));
}

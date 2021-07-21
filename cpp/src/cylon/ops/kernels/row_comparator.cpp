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

#include <cylon/ops/kernels/row_comparator.hpp>

bool cylon::kernel::RowComparator::operator()(const std::pair<int32_t, int64_t> &record1,
                                              const std::pair<int32_t, int64_t> &record2) const {
  return comparator->compare(tables->at(record1.first), record1.second,
                            tables->at(record2.first), record2.second) == 0;
}

size_t cylon::kernel::RowComparator::operator()(const std::pair<int32_t, int64_t> &record) const {
  return row_hashing_kernel->Hash(tables->at(record.first), record.second);
}

cylon::kernel::RowComparator::RowComparator(const std::shared_ptr<std::vector<std::shared_ptr<arrow::Table>>> &tables,
                                            const std::shared_ptr<arrow::Schema> &schema)
    : tables(tables),
      comparator(std::make_shared<TableRowComparator>(schema->fields())),
      row_hashing_kernel(std::make_shared<RowHashingKernel>(schema->fields())) {}

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


#ifndef CYLON_SRC_CYLON_OPS_KERNELS_UNION_HPP_
#define CYLON_SRC_CYLON_OPS_KERNELS_UNION_HPP_
#include <vector>
#include <table.hpp>
#include <ops/kernels/row_comparator.hpp>

namespace cylon {
namespace kernel {
class Union {
 private:
  std::vector<std::shared_ptr<arrow::Table >> tables{};
  std::vector<std::shared_ptr<std::vector<int64_t>>> indices_from_tabs{};
  std::shared_ptr<arrow::Schema> schema;
  shared_ptr<CylonContext> ctx;

  std::unordered_set<std::pair<int8_t, int64_t>, row_comparator, row_comparator> *rows_set;

 public:
  ~Union();
  Union(std::shared_ptr<cylon::CylonContext> ctx,
        std::shared_ptr<arrow::Schema> schema,
        int64_t expected_rows);
  void InsertTable(std::shared_ptr<cylon::Table> table);
  cylon::Status Finalize(std::shared_ptr<cylon::Table> &result);
};
}
}
#endif //CYLON_SRC_CYLON_OPS_KERNELS_UNION_HPP_

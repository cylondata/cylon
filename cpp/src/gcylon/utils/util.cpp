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

#include <gcylon/utils/util.hpp>

#include <cudf/sorting.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/reduction.hpp>
#include <cudf/aggregation.hpp>
#include <cudf/scalar/scalar.hpp>

namespace gcylon {

bool table_equal_with_sorting(cudf::table_view &tv1, cudf::table_view &tv2) {
  std::unique_ptr<cudf::table> sorted_table1 = cudf::sort(tv1);
  auto sorted_tv1 = sorted_table1->view();

  std::unique_ptr<cudf::table> sorted_table2 = cudf::sort(tv2);
  auto sorted_tv2 = sorted_table2->view();
  return table_equal(sorted_tv1, sorted_tv2);
}

bool table_equal(const cudf::table_view &tv1, const cudf::table_view &tv2) {
  if (tv1.num_columns() != tv2.num_columns()) {
    return false;
  } else if (tv1.num_rows() != tv2.num_rows()) {
    return false;
  }

  // whether the table columns have the same data type
  if (!cudf::have_same_types(tv1, tv2)) {
    return false;
  }

  // if the tables have zero rows
  if (tv1.num_rows() == 0) {
    return true;
  }

  auto agg = cudf::make_all_aggregation<cudf::reduce_aggregation>();
  cudf::data_type bool_type{cudf::type_id::BOOL8};

  // compare all elements in the table
  for (int i = 0; i < tv1.num_columns(); ++i) {
    const auto &result_column = cudf::binary_operation(tv1.column(i),
                                                       tv2.column(i),
                                                       cudf::binary_operator::NULL_EQUALS,
                                                       bool_type);
    auto col_view = result_column->view();
    std::unique_ptr<cudf::scalar> all = cudf::reduce(col_view, *agg, bool_type);
    std::unique_ptr<cudf::numeric_scalar<bool>> all_numeric(
        static_cast<cudf::numeric_scalar<bool> *>(all.release()));
    if (!all_numeric->value()) {
      return false;
    }
  }

  return true;
}

std::vector<cudf::table_view> tablesToViews(const std::vector<std::unique_ptr<cudf::table>> &tables) {
  std::vector<cudf::table_view> views;
  views.reserve(tables.size());
  std::transform(tables.begin(), tables.end(), std::back_inserter(views),
                 [](const std::unique_ptr<cudf::table>& t){ return t->view();});
  return views;
}

}// end of namespace gcylon

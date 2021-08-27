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

bool equal(cudf::table_view & tv1, cudf::table_view & tv2) {
    std::unique_ptr<cudf::table> sorted_table1 = cudf::sort(tv1);
    auto sorted_tv1 = sorted_table1->view();

    std::unique_ptr<cudf::table> sorted_table2 = cudf::sort(tv2);
    auto sorted_tv2 = sorted_table2->view();

    if (sorted_tv1.num_columns() != sorted_tv2.num_columns()){
        return false;
    } else if (sorted_tv1.num_rows() != sorted_tv2.num_rows()) {
        return false;
    }

    // whether the table columns have the same data type
    if (!cudf::have_same_types(tv1, tv2)) {
        return false;
    }

    std::unique_ptr<cudf::aggregation> agg = cudf::make_all_aggregation();
    cudf::data_type bool_type = cudf::data_type(cudf::type_id::BOOL8);

    // compare all elements in the table
    for (int i = 0; i < sorted_tv1.num_columns(); ++i) {
        std::unique_ptr<cudf::column> result_column = cudf::binary_operation(sorted_tv1.column(i),
                                                                            sorted_tv2.column(i),
                                                                            cudf::binary_operator::EQUAL,
                                                                            bool_type);
        std::unique_ptr<cudf::scalar> all = cudf::reduce(result_column->view(), agg, bool_type);
        std::unique_ptr<cudf::numeric_scalar<bool>> all_numeric(static_cast<cudf::numeric_scalar<bool> *>(all.release()));
        if (!all_numeric->value())
            return false;
    }

    return true;
}

}// end of namespace gcylon

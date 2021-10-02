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

#ifndef GCYLON_SORTING_SORTING_HPP
#define GCYLON_SORTING_SORTING_HPP

#include <memory>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cylon/net/buffer.hpp>

namespace gcylon {

/**
 * sample a table uniformly
 * @param tv
 * @param sample_count number of rows to have in the resulting table
 * @param sampled_table a new table with sampled rows only
 * @return
 */
cylon::Status SampleTableUniform(const cudf::table_view &tv,
                                 int sample_count,
                                 std::unique_ptr<cudf::table> &sampled_table);

/**
 * Get split points to all workers
 *      Splitter performs gather
 *      Sorts/merges gathered tables
 *      Determines split points
 *      Broadcasts split points to all workers
 * @param data_buffer
 * @param mask_buffer
 * @param offsets_buffer
 * @param dt
 * @param num_rows
 * @return
 */
cylon::Status GetSplitPoints(const cudf::table_view & sample_tv,
                             int splitter,
                             const std::vector<cudf::order> &column_orders,
                             const std::shared_ptr<cylon::CylonContext> &ctx,
                             std::unique_ptr<cudf::table> &split_table);

} // end of namespace gcylon

#endif //GCYLON_SORTING_SORTING_HPP

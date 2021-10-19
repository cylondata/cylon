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
 * get the sampling ratio for sorting
 * when the number of workers is small, we sample more:
 * because it does not have much performance penalty sampling more data when the number of workers is small
 * and it gives better result with more balanced data distribution after sorting
 * @param num_workers
 * @return
 */
inline int GetSamplingRatio(int num_workers) {
  return num_workers < 100 ? 2 : 1;
}

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
cylon::Status GetSplitPoints(std::unique_ptr<cudf::table> sample_tbl,
                             int splitter,
                             const std::vector<cudf::order> &column_orders,
                             cudf::null_order null_ordering,
                             const std::shared_ptr<cylon::CylonContext> &ctx,
                             std::unique_ptr<cudf::table> &split_table);

/**
 * determine whether sorting or merging be performed to produce a single sorted table
 * from many sorted tables
 * when there are a lot of tables to merge,
 * sorting may be faster
 * We determined these parameters experimentally
 * We merged and sorted tables in total size of 1GB
 * @param data_size
 * @param num_columns
 * @return true for merge and false for sort
 */
bool MergeOrSort(int num_columns, int num_tables);

} // end of namespace gcylon

#endif //GCYLON_SORTING_SORTING_HPP

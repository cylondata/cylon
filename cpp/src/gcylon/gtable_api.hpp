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

#ifndef GCYLON_GTABLE_API_H
#define GCYLON_GTABLE_API_H

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/io/types.hpp>

#include <gcylon/gtable.hpp>
#include <cylon/join/join_config.hpp>

namespace gcylon {

/**
 * Shuffles a cudf::table with table_view
 * this is to be called from cython code and the other Shuffle with GTable
 * @param input_tv
 * @param columns_to_hash
 * @param ctx
 * @param table_out
 * @return
 */
cylon::Status Shuffle(const cudf::table_view &input_tv,
                      const std::vector<int> &columns_to_hash,
                      const std::shared_ptr<cylon::CylonContext> &ctx,
                      std::unique_ptr<cudf::table> &table_out);

/**
 * Repartition the table by either evenly distributing rows among all workers
 *   or according to the partition map given by rows_per_worker
 * @param input_tv
 * @param ctx
 * @param table_out
 * @param rows_per_worker
 * @return
 */
cylon::Status Repartition(const cudf::table_view &input_tv,
                          const std::shared_ptr<cylon::CylonContext> &ctx,
                          std::unique_ptr<cudf::table> &table_out,
                          const std::vector<int32_t> &rows_per_worker = std::vector<int32_t>());

/**
 * Similar to local join, but performs the join in a distributed fashion
 * @param left_table
 * @param right_table
 * @param join_config
 * @param ctx
 * @param table_out
 * @return <cylon::Status>
 */
cylon::Status DistributedJoin(const cudf::table_view & left_table,
                              const cudf::table_view & right_table,
                              const cylon::join::config::JoinConfig &join_config,
                              const std::shared_ptr<cylon::CylonContext> &ctx,
                              std::unique_ptr<cudf::table> &table_out);


/**
* Shuffles a GTable based on hashes of the given columns
* @param table
* @param hash_col_idx vector of column indicies that needs to be hashed
* @param output
* @return
*/
cylon::Status Shuffle(std::shared_ptr<GTable> &input_table,
                      const std::vector<int> &columns_to_hash,
                      std::shared_ptr<GTable> &outputTable);

/**
 * Similar to local join, but performs the join in a distributed fashion
 * @param left
 * @param right
 * @param join_config
 * @param output
 * @return <cylon::Status>
 */
cylon::Status DistributedJoin(std::shared_ptr<GTable> &left,
                       std::shared_ptr<GTable> &right,
                       const cylon::join::config::JoinConfig &join_config,
                       std::shared_ptr<GTable> &output);

/**
 * write the csv table to a file
 * @param table
 * @param output_file
 * @return
 */
cylon::Status WriteToCsv(std::shared_ptr<GTable> &table, std::string output_file);

/**
 * perform distributed sort on provided table
 * @param tv
 * @param sort_column_indices sort based on these columns
 * @param ctx
 * @param sorted_table resulting table
 * @param sort_root the worker that will determine the global split points
 * @param sort_ascending
 * @param nulls_after
 * @return
 */
cylon::Status DistributedSort(const cudf::table_view &tv,
                              const std::vector<int32_t> &sort_column_indices,
                              const std::vector<cudf::order> &column_orders,
                              const std::shared_ptr<cylon::CylonContext> &ctx,
                              std::unique_ptr<cudf::table> &sorted_table,
                              bool nulls_after = true,
                              const int sort_root = 0);

/**
 * get table sizes from all workers
 * each worker size in the table_sizes[rank]
 * @param num_rows size of the table at the current worker
 * @param ctx
 * @param all_num_rows all tables sizes from all workers
 * @return
 */
cylon::Status RowCountsAllTables(int32_t num_rows,
                                 const std::shared_ptr<cylon::CylonContext> &ctx,
                                 std::vector<int32_t> &all_num_rows);


}// end of namespace gcylon

#endif //GCYLON_GTABLE_API_H

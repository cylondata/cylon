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

#ifndef GCYLON_CUDF_NET_OPS_HPP
#define GCYLON_CUDF_NET_OPS_HPP

#include <cudf/table/table.hpp>
#include <cylon/status.hpp>
#include <cylon/ctx/cylon_context.hpp>

namespace gcylon {
namespace net {

/**
 * calculate number of rows in received buffers
 * @param received_buffers received buffers over the wire, encoded with cylon::TableSerializer::getDataBuffers()
 * @param data_types data type of the first column in the table, encoded with cylon::TableSerializer::getDataTypes()
 * @return
 */
int32_t numOfRows(const std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                  const int32_t data_type);

/**
 * calculate the number of rows in a column
 * @param data_types data type of the column in the table
 * @param data_size size of the data in the column in bytes
 * @param offsets_size size of the offsets buffer int the column in bytes
 * @return
 */
int32_t numOfRows(const cudf::data_type dt,
                  const int32_t data_size,
                  const int32_t offsets_size);

/**
 * Broadcast a table to all workers except the broadcast root
 * @param tv the table to be broadcasted (significant at broadcast root only)
 * @param bcast_root broadcaster worker ID (significant at all workers)
 * @param ctx Cylon context (significant at all workers)
 * @param received_table all workers except the broadcast root get a table (significant at all receiver workers)
 * @return
 */
cylon::Status Bcast(const cudf::table_view &tv,
                    const int bcast_root,
                    const std::shared_ptr<cylon::CylonContext> ctx,
                    std::unique_ptr<cudf::table> &received_table);

/**
 * Gather CuDF tables
 * @param tv the table to gather
 * @param gather_root MPI rank of the worker that gathers tables
 * @param gather_from_root Whether the table from the root will be gathered
 * @param ctx Cylon context
 * @param gathered_tables gathered tables at the gather root
 * @return
 */
cylon::Status Gather(const cudf::table_view &tv,
                     const int gather_root,
                     bool gather_from_root,
                     std::shared_ptr<cylon::CylonContext> ctx,
                     std::vector<std::unique_ptr<cudf::table>> &gathered_tables);


/**
 * MPI All to all for a CuDF Table on each worker
 * Each table has n partitions for n workers
 * part_indices has (n+1) partition indices each range for a worker
 *   range(i, i+1) goes to the worker[i]
 *
 * received tables put in the resulting vector in order of worker rank:
 *   received table from worker 0 is at received_tables[0]
 *   received table from worker 1 is at received_tables[1]
 *   ...
 *
 *  if a worker does not send any table to another another worker,
 *    an empty table is put in that worker rank in received_tables vector
 *
 * received_tables has exactly n tables (some tables might be empty though)
 *
 * @param tv table_view with partitions
 * @param part_indices (n-1) partition indices
 * @param ctx
 * @param received_tables
 * @return
 */
cylon::Status AllToAll(const cudf::table_view & tv,
                       const std::vector<cudf::size_type> &part_indices,
                       const std::shared_ptr<cylon::CylonContext> &ctx,
                       std::vector<std::unique_ptr<cudf::table>> &received_tables);

/**
 * The same as the previous AllToAll except that
 * it concatenates all received tables and return a single table
 * @param tv
 * @param part_indices
 * @param ctx
 * @param table_out
 * @return
 */
cylon::Status AllToAll(const cudf::table_view &tv,
                       const std::vector<cudf::size_type> &part_indices,
                       const std::shared_ptr<cylon::CylonContext> &ctx,
                       std::unique_ptr<cudf::table> &table_out);

} // end of namespace net
} // end of namespace gcylon

#endif //GCYLON_CUDF_NET_OPS_HPP

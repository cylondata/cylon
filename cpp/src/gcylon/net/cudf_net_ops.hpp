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
 * whether this worker is the root
 * @param ctx
 * @param root
 * @return
 */
bool Am_I_Root(std::shared_ptr<cylon::CylonContext> ctx, const int root);

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
cylon::Status Bcast(cudf::table_view &tv,
                    const int bcast_root,
                    const std::shared_ptr<cylon::CylonContext> ctx,
                    std::unique_ptr<cudf::table> &received_table);

} // end of namespace net
} // end of namespace gcylon

#endif //GCYLON_CUDF_NET_OPS_HPP

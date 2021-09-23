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

#include <cylon/util/macros.hpp>
#include <cylon/net/mpi/mpi_operations.hpp>
#include <gcylon/net/cudf_net_ops.hpp>
#include <gcylon/cudf_buffer.hpp>
#include <gcylon/sorting/deserialize.hpp>
#include <gcylon/net/cudf_serialize.hpp>

cylon::Status gcylon::net::Bcast(cudf::table_view &tv,
                   const int bcast_root,
                   const std::shared_ptr<cylon::CylonContext> ctx,
                   std::unique_ptr<cudf::table> &received_table) {

    std::shared_ptr<CudfTableSerializer> serializer;
    if (cylon::mpi::AmIRoot(bcast_root, ctx)) {
        serializer = std::make_shared<CudfTableSerializer>(tv);
    }
    auto allocator = std::make_shared<CudfAllocator>();
    std::vector<std::shared_ptr<cylon::Buffer>> received_buffers;
    std::vector<int32_t> data_types;

    RETURN_CYLON_STATUS_IF_FAILED(
            cylon::mpi::Bcast(serializer, bcast_root, allocator, received_buffers, data_types, ctx));

    if (!cylon::mpi::AmIRoot(bcast_root, ctx)){
        RETURN_CYLON_STATUS_IF_FAILED(
                gcylon::deserializeSingleTable(received_buffers, data_types, received_table));
    }

    return cylon::Status::OK();
}

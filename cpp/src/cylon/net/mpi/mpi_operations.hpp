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

#ifndef CYLON_CPP_SRC_CYLON_NET_MPI_MPI_OPERATIONS_HPP_
#define CYLON_CPP_SRC_CYLON_NET_MPI_MPI_OPERATIONS_HPP_

#include <mpi.h>
#include <memory>
#include <cylon/net/serialize.hpp>
#include <cylon/net/buffer.hpp>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/net/comm_operations.hpp>

namespace cylon {
namespace mpi {

MPI_Op GetMPIOp(cylon::net::ReduceOp reduce_op);

MPI_Datatype GetMPIDataType(const std::shared_ptr<DataType> &data_type);

cylon::Status AllReduce(const void *send_buf,
                        void *rcv_buf,
                        int count,
                        const std::shared_ptr<DataType> &data_type,
                        cylon::net::ReduceOp reduce_op);

/**
 * check whether the invoking worker is the root
 * @param root
 * @param ctx
 * @return
 */
bool AmIRoot(int root, const std::shared_ptr<cylon::CylonContext>& ctx);

/**
 * Perform MPI Gather on a table
 * @param serializer TableSerializer to serialize a table
 * @param gather_root MPI rank of the gather root worker
 * @param gather_from_root whether the table will be gathered from the root also
 * @param allocator Allocator to allocate the buffer for received data
 * @param all_buffer_sizes all received buffer size (significant at gather root only)
 * @param received_buffers all received buffers (significant at gather root only)
 * @param displacements displacements in each buffer (significant at gather root only)
 * @param ctx CylonContext object
 * @return
 */
cylon::Status Gather(std::shared_ptr<cylon::TableSerializer> serializer,
                     const int gather_root,
                     const bool gather_from_root,
                     std::shared_ptr<cylon::Allocator> allocator,
                     std::unique_ptr<int32_t []> & all_buffer_sizes,
                     std::vector<std::shared_ptr<cylon::Buffer>> & received_buffers,
                     std::vector<std::vector<int32_t>> & displacements,
                     std::shared_ptr<cylon::CylonContext> ctx
);

/**
 * Perform MPI broadcast on a table
 * @param serializer TableSerializer to serialize a table (significant at broadcast root only)
 * @param bcast_root MPI rank of the broadcaster worker, (significant at all workers)
 * @param allocator Allocator to allocate the received buffers (significant at all receiver workers)
 * @param received_buffers all received buffers (significant at all receiver workers)
 * @param data_types data types of the table column (significant at all receiver workers)
 * @param ctx CylonContext object (significant at all workers)
 * @return
 */
cylon::Status Bcast(std::shared_ptr<cylon::TableSerializer> serializer,
                    const int bcast_root,
                    std::shared_ptr<cylon::Allocator> allocator,
                    std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                    std::vector<int32_t> &data_types,
                    std::shared_ptr<cylon::CylonContext> ctx
);

}
}
#endif //CYLON_CPP_SRC_CYLON_NET_MPI_MPI_OPERATIONS_HPP_

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

#ifndef CYLON_NET_OPS_BCAST_HPP_
#define CYLON_NET_OPS_BCAST_HPP_

#include <memory>
#include <cylon/net/serialize.hpp>
#include <cylon/net/buffer.hpp>
#include <cylon/ctx/cylon_context.hpp>

namespace cylon {
namespace mpi {

/**
 *
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
#endif //CYLON_NET_OPS_BCAST_HPP_

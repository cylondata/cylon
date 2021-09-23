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

#include <cylon/net/mpi/mpi_operations.hpp>

cylon::Status cylon::mpi::Bcast(std::shared_ptr<cylon::TableSerializer> serializer,
                                const int bcast_root,
                                std::shared_ptr<cylon::Allocator> allocator,
                                std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                                std::vector<int32_t> &data_types,
                                std::shared_ptr<cylon::CylonContext> ctx) {


    // first broadcast the number of buffers
    int32_t num_buffers = 0;
    if(AmIRoot(bcast_root, ctx)) {
        num_buffers = serializer->getNumberOfBuffers();
    }

    int status = MPI_Bcast(&num_buffers, 1, MPI_INT32_T, bcast_root, MPI_COMM_WORLD);
    if (status != MPI_SUCCESS) {
        return cylon::Status(cylon::Code::ExecutionError, "MPI_Bcast failed for size broadcast!");
    }

    // broadcast buffer sizes
    std::vector<int32_t> buffer_sizes(num_buffers, 0);
    if(AmIRoot(bcast_root, ctx)) {
        buffer_sizes = serializer->getBufferSizes();
    }

    status = MPI_Bcast(buffer_sizes.data(), num_buffers, MPI_INT32_T, bcast_root, MPI_COMM_WORLD);
    if (status != MPI_SUCCESS) {
        return cylon::Status(cylon::Code::ExecutionError, "MPI_Bcast failed for size araay broadcast!");
    }

    // broadcast data types
    if(AmIRoot(bcast_root, ctx)) {
        data_types = serializer->getDataTypes();
    } else {
        data_types.resize(num_buffers / 3, 0);
    }

    status = MPI_Bcast(data_types.data(), data_types.size(), MPI_INT32_T, bcast_root, MPI_COMM_WORLD);
    if (status != MPI_SUCCESS) {
        return cylon::Status(cylon::Code::ExecutionError, "MPI_Bcast failed for data type array broadcast!");
    }

    auto requests = std::make_unique<MPI_Request []>(num_buffers);
    auto statuses = std::make_unique<MPI_Status []>(num_buffers);
    std::vector<uint8_t *> send_buffers{};
    if (AmIRoot(bcast_root, ctx)) {
        send_buffers = serializer->getDataBuffers();
    }

    for (int32_t i = 0; i < num_buffers; ++i) {
        if(AmIRoot(bcast_root, ctx)) {
            if (buffer_sizes[i] == 0) {
                continue;
            }
            status = MPI_Bcast(send_buffers.at(i),
                                buffer_sizes[i],
                                MPI_UINT8_T,
                                bcast_root,
                                MPI_COMM_WORLD);
            if (status != MPI_SUCCESS) {
                return cylon::Status(cylon::Code::ExecutionError, "MPI_Ibast failed!");
            }
        } else {
            std::shared_ptr<cylon::Buffer> receive_buf;
            allocator->Allocate(buffer_sizes[i], &receive_buf);
            if (buffer_sizes[i] == 0) {
                received_buffers.push_back(receive_buf);
                continue;
            }

            status = MPI_Bcast(receive_buf->GetByteBuffer(),
                                buffer_sizes[i],
                                MPI_UINT8_T,
                                bcast_root,
                                MPI_COMM_WORLD);
            if (status != MPI_SUCCESS) {
                return cylon::Status(cylon::Code::ExecutionError, "MPI_Ibast failed!");
            }
            received_buffers.push_back(receive_buf);
        }
    }

//    status = MPI_Waitall(num_buffers, requests.get(), statuses.get());
//    if (status != MPI_SUCCESS) {
//        return cylon::Status(cylon::Code::ExecutionError, "MPI_Ibast failed when waiting!");
//    }

    return cylon::Status::OK();
}

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

#include <mpi.h>
#include <cylon/net/mpi/mpi_operations.hpp>


bool cylon::mpi::AmIRoot(int root, std::shared_ptr<cylon::CylonContext> ctx) {
    return root == ctx->GetRank();
}

std::vector<int32_t> totalBufferSizes(int32_t *all_buffer_sizes, int num_buffers, int world_size) {
    std::vector<int32_t> total_buffer_sizes(num_buffers, 0);
    for (int i = 0; i < num_buffers; ++i) {
        for (int j = 0, k = i; j < world_size; ++j) {
            total_buffer_sizes[i] += all_buffer_sizes[k];
            k += num_buffers;
        }
    }
    return total_buffer_sizes;
}

std::vector<int32_t> receiveCounts(int32_t *all_buffer_sizes, int receiveNo, int num_buffers, int world_size) {
    std::vector<int32_t> receive_counts(world_size, 0);
    for (int i = 0, k = receiveNo; i < world_size; ++i) {
        receive_counts[i] = all_buffer_sizes[k];
        k += num_buffers;
    }
    return receive_counts;
}

std::vector<int32_t> displacementsPerBuffer(int32_t *all_buffer_sizes, int receiveNo, int num_buffers, int world_size) {
    std::vector<int32_t> disp_array(world_size, 0);
    disp_array[0] = 0;
    for (int i = 1, k = receiveNo; i < world_size; ++i) {
        disp_array[i] = disp_array[i-1] + all_buffer_sizes[k];
        k += num_buffers;
    }
    return disp_array;
}


cylon::Status cylon::mpi::Gather(std::shared_ptr<cylon::TableSerializer> serializer,
                     const int gather_root,
                     const bool gather_from_root,
                     std::shared_ptr<cylon::Allocator> allocator,
                     std::unique_ptr<int32_t []> & all_buffer_sizes,
                     std::vector<std::shared_ptr<cylon::Buffer>> & receive_buffers,
                     std::vector<std::vector<int32_t>> & displacements,
                     std::shared_ptr<cylon::CylonContext> ctx
                     ){

    // first gather table buffer sizes
    std::vector<int32_t> local_buffer_sizes;
    if(AmIRoot(gather_root, ctx) && !gather_from_root) {
        local_buffer_sizes = serializer->getEmptyTableBufferSizes();
    } else {
        local_buffer_sizes = serializer->getBufferSizes();
    }

    int32_t num_buffers = local_buffer_sizes.size();

    // gather size buffers
    if (AmIRoot(gather_root, ctx)) {
        all_buffer_sizes = std::make_unique<int32_t []>(ctx->GetWorldSize() * num_buffers);
    }

    int status = MPI_Gather(local_buffer_sizes.data(),
                            num_buffers,
                            MPI_INT32_T,
                            all_buffer_sizes.get(),
                            num_buffers,
                            MPI_INT32_T,
                            gather_root,
                            MPI_COMM_WORLD);
    if (status != MPI_SUCCESS) {
        return cylon::Status(cylon::Code::ExecutionError, "MPI_Gather failed!");
    }

    std::vector<int32_t> total_buffer_sizes;
    if (AmIRoot(gather_root, ctx)) {
        totalBufferSizes(all_buffer_sizes.get(), num_buffers, ctx->GetWorldSize()).swap(total_buffer_sizes);
    }

    auto requests = std::make_unique<MPI_Request []>(num_buffers);
    auto statuses = std::make_unique<MPI_Status []>(num_buffers);
    std::vector<uint8_t *> send_buffers = serializer->getDataBuffers();

    for (int32_t i = 0; i < num_buffers; ++i) {
        if(AmIRoot(gather_root, ctx)) {
            std::shared_ptr<cylon::Buffer> receive_buf;
            allocator->Allocate(total_buffer_sizes[i], &receive_buf);
            std::vector<int32_t> receive_counts = receiveCounts(all_buffer_sizes.get(),
                                                                i,
                                                                num_buffers,
                                                                ctx->GetWorldSize());
            std::vector<int32_t> disp_per_buffer = displacementsPerBuffer(all_buffer_sizes.get(),
                                                                          i,
                                                                          num_buffers,
                                                                          ctx->GetWorldSize());
            displacements.push_back(disp_per_buffer);

            status = MPI_Igatherv(send_buffers.at(i),
                                  local_buffer_sizes[i],
                                  MPI_UINT8_T,
                                  receive_buf->GetByteBuffer(),
                                  receive_counts.data(),
                                  disp_per_buffer.data(),
                                  MPI_UINT8_T,
                                  gather_root,
                                  MPI_COMM_WORLD,
                                  &requests[i]);
            if (status != MPI_SUCCESS) {
                return cylon::Status(cylon::Code::ExecutionError, "MPI_Igatherv failed!");
            }
            receive_buffers.push_back(receive_buf);

        } else {
            status = MPI_Igatherv(send_buffers.at(i),
                                  local_buffer_sizes[i],
                                  MPI_UINT8_T,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  MPI_UINT8_T,
                                  gather_root,
                                  MPI_COMM_WORLD,
                                  &requests[i]);
            if (status != MPI_SUCCESS) {
                return cylon::Status(cylon::Code::ExecutionError, "MPI_Igatherv failed!");
            }
        }
    }

    status = MPI_Waitall(num_buffers, requests.get(), statuses.get());
    if (status != MPI_SUCCESS) {
        return cylon::Status(cylon::Code::ExecutionError, "MPI_Igatherv failed!");
    }

    return cylon::Status::OK();
}



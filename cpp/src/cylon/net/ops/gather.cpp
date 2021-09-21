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
#include <memory>
#include <cylon/net/ops/gather.hpp>


bool AmIRoot(int gather_root, std::shared_ptr<cylon::CylonContext> ctx) {
    return gather_root == ctx->GetRank();
}

std::vector<int32_t> totalBufferSizes(int32_t *all_buffer_sizes, int buffer_size_pw, int world_size) {
    std::vector<int32_t> total_buffer_sizes(buffer_size_pw, 0);
    for (int i = 0; i < buffer_size_pw; ++i) {
        for (int j = 0, k = i; j < world_size; ++j) {
            total_buffer_sizes[i] += all_buffer_sizes[k];
            k += buffer_size_pw;
        }
    }
    return total_buffer_sizes;
}

std::vector<int32_t> receiveCounts(int32_t *all_buffer_sizes, int receiveNo, int buffer_size_pw, int world_size) {
    std::vector<int32_t> receive_counts(world_size, 0);
    for (int i = 0, k = receiveNo; i < world_size; ++i) {
        receive_counts[i] = all_buffer_sizes[k];
        k += buffer_size_pw;
    }
    return receive_counts;
}

std::vector<int32_t> displacementsPerBuffer(int32_t *all_buffer_sizes, int receiveNo, int buffer_size_pw, int world_size) {
    std::vector<int32_t> disp_array(world_size, 0);
    disp_array[0] = 0;
    for (int i = 1, k = receiveNo; i < world_size; ++i) {
        disp_array[i] = disp_array[i-1] + all_buffer_sizes[k];
        k += buffer_size_pw;
    }
    return disp_array;
}


cylon::Status cylon::mpi::Gather(std::shared_ptr<cylon::TableSerializer> serializer,
                     int gather_root,
                     bool gather_from_root,
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

    // gather size buffers
    if (AmIRoot(gather_root, ctx)) {
        all_buffer_sizes = std::make_unique<int32_t []>(ctx->GetWorldSize() * local_buffer_sizes.size());
    }

    int status = MPI_Gather(local_buffer_sizes.data(),
                            local_buffer_sizes.size(),
                            MPI_INT32_T,
                            all_buffer_sizes.get(),
                            local_buffer_sizes.size(),
                            MPI_INT32_T,
                            gather_root,
                            MPI_COMM_WORLD);
    if (status != MPI_SUCCESS) {
        return cylon::Status(cylon::Code::ExecutionError, "MPI_Gather failed!");
    }

    std::vector<int32_t> total_buffer_sizes;
    if (AmIRoot(gather_root, ctx)) {
        totalBufferSizes(all_buffer_sizes.get(), local_buffer_sizes.size(), ctx->GetWorldSize()).swap(total_buffer_sizes);
    }

    std::vector<uint8_t *> send_buffers = serializer->getDataBuffers();
    for (long unsigned int i = 0; i < local_buffer_sizes.size(); ++i) {
        if(AmIRoot(gather_root, ctx)) {
            std::shared_ptr<cylon::Buffer> receiveBuf;
            allocator->Allocate(total_buffer_sizes[i], &receiveBuf);
            std::vector<int32_t> receive_counts = receiveCounts(all_buffer_sizes.get(), i, local_buffer_sizes.size(), ctx->GetWorldSize());
            std::vector<int32_t> disp_per_buffer = displacementsPerBuffer(all_buffer_sizes.get(), i, local_buffer_sizes.size(), ctx->GetWorldSize());
            displacements.push_back(disp_per_buffer);

            int status = MPI_Gatherv(send_buffers.at(i),
                                     local_buffer_sizes[i],
                                     MPI_UINT8_T,
                                     receiveBuf->GetByteBuffer(),
                                     receive_counts.data(),
                                     disp_per_buffer.data(),
                                     MPI_UINT8_T,
                                     gather_root,
                                     MPI_COMM_WORLD);
            if (status != MPI_SUCCESS) {
                return cylon::Status(cylon::Code::ExecutionError, "MPI_Gatherv failed!");
            }
            receive_buffers.push_back(receiveBuf);

        } else {
            int status = MPI_Gatherv(send_buffers.at(i),
                                     local_buffer_sizes[i],
                                     MPI_UINT8_T,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     MPI_UINT8_T,
                                     gather_root,
                                     MPI_COMM_WORLD);
            if (status != MPI_SUCCESS) {
                return cylon::Status(cylon::Code::ExecutionError, "MPI_Gatherv failed!");
            }
        }
    }

    return cylon::Status::OK();
}



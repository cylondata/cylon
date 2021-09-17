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

#include <glog/logging.h>
#include <gcylon/sorting/cudf_gather.hpp>
#include <gcylon/sorting/deserialize.hpp>

#include <mpi.h>
#include <cudf/table/table.hpp>

namespace gcylon {

TableGatherer::TableGatherer(std::shared_ptr<cylon::CylonContext> ctx,
                             const int gather_root,
                             std::shared_ptr<cylon::Allocator> allocator)
    : ctx_(ctx), root_(gather_root), allocator_(allocator) {
}

bool TableGatherer::AmIRoot() {
    return root_ == ctx_->GetRank();
}

void printTensor(std::vector<std::vector<int32_t>> & tensor) {
    std::cout << "Tensor: " << std::endl;
    for (long unsigned int i = 0; i < tensor.size(); ++i) {
        std::cout << "line [" << i << "]: ";
        for (long unsigned int j = 0; j < tensor.at(i).size(); ++j) {
            std::cout << tensor.at(i).at(j) << ", ";
        }
        std::cout << std::endl;
    }
}


void printBufferSizes(int32_t * all_buffer_sizes, int buffer_size_pw, int workers) {
    std::cout << "buffer sizes: " << std::endl;
    int index = 0;
    for (int i = 0; i < workers; ++i) {
        std::cout << "worker [" << i << "]: ";
        for (int j = 0; j < buffer_size_pw; ++j) {
            std::cout << all_buffer_sizes[index++] << ", ";
        }
        std::cout << std::endl;
    }
}

void printBuffer(std::vector<int32_t> & buffer) {
    std::cout << "buffer: " << std::endl;
    for (long unsigned int j = 0; j < buffer.size(); ++j) {
        std::cout << buffer[j] << ", ";
    }
    std::cout << std::endl;
}

std::vector<int32_t> TableGatherer::totalBufferSizes(int32_t *all_buffer_sizes, int buffer_size_pw) {
    std::vector<int32_t> total_buffer_sizes(buffer_size_pw, 0);
    for (int i = 0; i < buffer_size_pw; ++i) {
        for (int j = 0, k = i; j < ctx_->GetWorldSize(); ++j) {
            total_buffer_sizes[i] += all_buffer_sizes[k];
            k += buffer_size_pw;
        }
    }
    return total_buffer_sizes;
}

std::vector<int32_t> TableGatherer::receiveCounts(int32_t *all_buffer_sizes, int receiveNo, int buffer_size_pw) {
    std::vector<int32_t> receive_counts(ctx_->GetWorldSize(), 0);
    for (int i = 0, k = receiveNo; i < ctx_->GetWorldSize(); ++i) {
        receive_counts[i] = all_buffer_sizes[k];
        k += buffer_size_pw;
    }
    return receive_counts;
}

std::vector<int32_t> TableGatherer::displacementsPerBuffer(int32_t *all_buffer_sizes, int receiveNo, int buffer_size_pw) {
    std::vector<int32_t> disp_array(ctx_->GetWorldSize(), 0);
    disp_array[0] = 0;
    for (int i = 1, k = receiveNo; i < ctx_->GetWorldSize(); ++i) {
        disp_array[i] = disp_array[i-1] + all_buffer_sizes[k];
        k += buffer_size_pw;
    }
    return disp_array;
}

std::vector<std::vector<int32_t>> TableGatherer::bufferSizesPerTable(int32_t *all_buffer_sizes, int buffer_size_pw) {
    std::vector<std::vector<int32_t>> buffer_sizes_all_tables;
    for (int i = 0, k = 0; i < ctx_->GetWorldSize(); ++i) {
        std::vector<int32_t> single_table_buffer_sizes;
        for (int j = 0; j < buffer_size_pw; ++j) {
            single_table_buffer_sizes.push_back(all_buffer_sizes[k++]);
        }
        buffer_sizes_all_tables.push_back(single_table_buffer_sizes);
    }
    return buffer_sizes_all_tables;
}


cylon::Status TableGatherer::Gather(cudf::table_view &tv,
                                    std::vector<std::unique_ptr<cudf::table>> &gathered_tables) {

    TableSerializer serializer(tv);

    // first gather table buffer sizes
    std::vector<int32_t> local_buffer_sizes;
    if(AmIRoot()) {
        local_buffer_sizes = serializer.getEmptyTableBufferSizes();
    } else {
        local_buffer_sizes = serializer.getBufferSizes();
    }

    // gather size buffers
    std::unique_ptr<int32_t []> all_buffer_sizes = nullptr;
    if (AmIRoot()) {
        all_buffer_sizes = std::make_unique<int32_t []>(ctx_->GetWorldSize() * local_buffer_sizes.size());
    }

    int status = MPI_Gather(local_buffer_sizes.data(),
                            local_buffer_sizes.size(),
                            MPI_INT32_T,
                            all_buffer_sizes.get(),
                            local_buffer_sizes.size(),
                            MPI_INT32_T,
                            root_,
                            MPI_COMM_WORLD);
    if (status != MPI_SUCCESS) {
        return cylon::Status(cylon::Code::ExecutionError, "MPI_Gather failed!");
    }

    std::vector<int32_t> total_buffer_sizes;
    if (AmIRoot()) {
        totalBufferSizes(all_buffer_sizes.get(), local_buffer_sizes.size()).swap(total_buffer_sizes);
    }

    //todo: delete
    if (AmIRoot()) {
        printBufferSizes(all_buffer_sizes.get(), local_buffer_sizes.size(), ctx_->GetWorldSize());
        printBuffer(total_buffer_sizes);
    }

    std::vector<uint8_t *> send_buffers = serializer.getDataBuffers();
    std::vector<std::shared_ptr<cylon::Buffer>> receive_buffers;
    std::vector<std::vector<int32_t>> all_disps;
    for (long unsigned int i = 0; i < local_buffer_sizes.size(); ++i) {
        if(AmIRoot()) {
            std::shared_ptr<cylon::Buffer> receiveBuf;
            allocator_->Allocate(total_buffer_sizes[i], &receiveBuf);
            std::vector<int32_t> receive_counts = receiveCounts(all_buffer_sizes.get(), i, local_buffer_sizes.size());
            std::vector<int32_t> disp_per_buffer = displacementsPerBuffer(all_buffer_sizes.get(), i, local_buffer_sizes.size());
            all_disps.push_back(disp_per_buffer);

            int status = MPI_Gatherv(send_buffers.at(i),
                                     local_buffer_sizes[i],
                                     MPI_UINT8_T,
                                     receiveBuf->GetByteBuffer(),
                                     receive_counts.data(),
                                     disp_per_buffer.data(),
                                     MPI_UINT8_T,
                                     root_,
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
                                     root_,
                                     MPI_COMM_WORLD);
            if (status != MPI_SUCCESS) {
                return cylon::Status(cylon::Code::ExecutionError, "MPI_Gatherv failed!");
            }
        }
    }

    if (AmIRoot()) {
        std::vector<std::vector<int32_t>> buffer_sizes_per_table =
                bufferSizesPerTable(all_buffer_sizes.get(), local_buffer_sizes.size());

        std::cout << "buffer sizes per table: " << std::endl;
        printTensor(buffer_sizes_per_table);

        std::cout << "displacements per buffer: " << std::endl;
        printTensor(all_disps);

        std::cout << "received buffer sizes: " << std::endl;
        for (long unsigned int i = 0; i < receive_buffers.size(); ++i) {
            std::cout << receive_buffers[i]->GetLength() << ", ";
        }
        std::cout << std::endl;

        TableDeserializer deserializer(tv);
        deserializer.deserialize(receive_buffers, all_disps, buffer_sizes_per_table, gathered_tables);
    }

    return cylon::Status::OK();
}


} // end of namespace gcylon
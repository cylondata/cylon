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
#include <cylon/net/ops/gather.hpp>
#include <gcylon/cudf_buffer.hpp>
#include <gcylon/sorting/cudf_gather.hpp>
#include <gcylon/sorting/deserialize.hpp>

#include <cudf/table/table.hpp>

namespace gcylon {

CudfTableGatherer::CudfTableGatherer(std::shared_ptr<cylon::CylonContext> ctx, const int gather_root)
    : ctx_(ctx), root_(gather_root) {
}

bool CudfTableGatherer::AmIRoot() {
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

std::vector<std::vector<int32_t>> CudfTableGatherer::bufferSizesPerTable(int32_t *all_buffer_sizes,
                                                                         int number_of_buffers) {
    std::vector<std::vector<int32_t>> buffer_sizes_all_tables;
    for (int i = 0, k = 0; i < ctx_->GetWorldSize(); ++i) {
        std::vector<int32_t> single_table_buffer_sizes;
        for (int j = 0; j < number_of_buffers; ++j) {
            single_table_buffer_sizes.push_back(all_buffer_sizes[k++]);
        }
        buffer_sizes_all_tables.push_back(single_table_buffer_sizes);
    }
    return buffer_sizes_all_tables;
}


cylon::Status CudfTableGatherer::Gather(cudf::table_view &tv,
                                    bool gather_from_root,
                                    std::vector<std::unique_ptr<cudf::table>> &gathered_tables) {

    auto serializer = std::make_shared<CudfTableSerializer>(tv);
    auto allocator = std::make_shared<CudfAllocator>();
    std::unique_ptr<int32_t []> all_buffer_sizes;
    std::vector<std::shared_ptr<cylon::Buffer>> receive_buffers;
    std::vector<std::vector<int32_t>> all_disps;

    cylon::Status status = cylon::mpi::Gather(serializer,
                                              root_,
                                              gather_from_root,
                                              allocator,
                                              all_buffer_sizes,
                                              receive_buffers,
                                              all_disps,
                                              ctx_);

    if (!status.is_ok()) {
        return status;
    }

    if (AmIRoot()) {
        std::vector<std::vector<int32_t>> buffer_sizes_per_table =
                bufferSizesPerTable(all_buffer_sizes.get(), receive_buffers.size());

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
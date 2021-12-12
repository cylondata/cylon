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
#include <algorithm>
#include <arrow/result.h>
#include <cylon/net/mpi/mpi_operations.hpp>
#include <cylon/util/macros.hpp>

cylon::Status cylon::mpi::Bcast(const std::shared_ptr<cylon::TableSerializer> &serializer,
                                int bcast_root,
                                const std::shared_ptr<cylon::Allocator> &allocator,
                                std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                                std::vector<int32_t> &data_types,
                                const std::shared_ptr<cylon::CylonContext> &ctx) {
  // first broadcast the number of buffers
  int32_t num_buffers = 0;
  if (AmIRoot(bcast_root, ctx)) {
    num_buffers = serializer->getNumberOfBuffers();
  }

  int status = MPI_Bcast(&num_buffers, 1, MPI_INT32_T, bcast_root, MPI_COMM_WORLD);
  if (status != MPI_SUCCESS) {
    return cylon::Status(cylon::Code::ExecutionError, "MPI_Bcast failed for size broadcast!");
  }

  // broadcast buffer sizes
  std::vector<int32_t> buffer_sizes(num_buffers, 0);
  if (AmIRoot(bcast_root, ctx)) {
    buffer_sizes = serializer->getBufferSizes();
  }

  status = MPI_Bcast(buffer_sizes.data(), num_buffers, MPI_INT32_T, bcast_root, MPI_COMM_WORLD);
  if (status != MPI_SUCCESS) {
    return cylon::Status(cylon::Code::ExecutionError, "MPI_Bcast failed for size array broadcast!");
  }

  // broadcast data types
  if (AmIRoot(bcast_root, ctx)) {
    data_types = serializer->getDataTypes();
  } else {
    data_types.resize(num_buffers / 3, 0);
  }

  status = MPI_Bcast(data_types.data(), data_types.size(), MPI_INT32_T, bcast_root, MPI_COMM_WORLD);
  if (status != MPI_SUCCESS) {
    return cylon::Status(cylon::Code::ExecutionError, "MPI_Bcast failed for data type array broadcast!");
  }

  // if all buffer sizes are zero, there are zero rows in the table
  // no need to broadcast any buffers
  if(std::all_of(buffer_sizes.begin(), buffer_sizes.end(), [](int32_t i) { return i == 0; })) {
    return cylon::Status::OK();
  }

  std::vector<MPI_Request> requests(num_buffers);
  std::vector<MPI_Status> statuses(num_buffers);
  std::vector<const uint8_t *> send_buffers{};
  if (AmIRoot(bcast_root, ctx)) {
    send_buffers = serializer->getDataBuffers();
  } else {
    received_buffers.reserve(num_buffers);
  }

  for (int32_t i = 0; i < num_buffers; ++i) {
    if (AmIRoot(bcast_root, ctx)) {
      status = MPI_Ibcast(const_cast<uint8_t *>(send_buffers[i]),
                          buffer_sizes[i],
                          MPI_UINT8_T,
                          bcast_root,
                          MPI_COMM_WORLD,
                          &requests[i]);
      if (status != MPI_SUCCESS) {
        return cylon::Status(cylon::Code::ExecutionError, "MPI_Ibast failed!");
      }
    } else {
      std::shared_ptr<cylon::Buffer> receive_buf;
      RETURN_CYLON_STATUS_IF_FAILED(allocator->Allocate(buffer_sizes[i], &receive_buf));

      status = MPI_Ibcast(receive_buf->GetByteBuffer(),
                          buffer_sizes[i],
                          MPI_UINT8_T,
                          bcast_root,
                          MPI_COMM_WORLD,
                          &requests[i]);
      if (status != MPI_SUCCESS) {
        return cylon::Status(cylon::Code::ExecutionError, "MPI_Ibast failed!");
      }
      received_buffers.push_back(receive_buf);
    }
  }

  status = MPI_Waitall(num_buffers, requests.data(), statuses.data());
  if (status != MPI_SUCCESS) {
    return cylon::Status(cylon::Code::ExecutionError, "MPI_Ibast failed when waiting!");
  }

  return cylon::Status::OK();
}

cylon::Status cylon::mpi::BcastArrowBuffer(std::shared_ptr<arrow::Buffer> &buf,
                                           int bcast_root,
                                           const std::shared_ptr<cylon::CylonContext> &ctx) {

  // first broadcast the buffer size
  int32_t buf_size = 0;
  if (AmIRoot(bcast_root, ctx)) {
    buf_size = static_cast<int32_t>(buf->size());
  }

  int status = MPI_Bcast(&buf_size, 1, MPI_INT32_T, bcast_root, MPI_COMM_WORLD);
  if (status != MPI_SUCCESS) {
    return cylon::Status(cylon::Code::ExecutionError, "MPI_Bcast failed for buf_size broadcast!");
  }

  // allocate arrow Buffers if not the root
  if (!AmIRoot(bcast_root, ctx)) {
    buf = std::move(arrow::AllocateBuffer(buf_size).MoveValueUnsafe());
  }

  status = MPI_Bcast((void *) buf->data(), buf_size, MPI_UINT8_T, bcast_root, MPI_COMM_WORLD);
  if (status != MPI_SUCCESS) {
    return cylon::Status(cylon::Code::ExecutionError, "MPI_Bcast failed for arrow::Buffer broadcast!");
  }

  return cylon::Status::OK();
}
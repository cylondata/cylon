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

cylon::mpi::MpiTableBcastImpl::MpiTableBcastImpl(MPI_Comm comm) : comm_(comm) {}

void cylon::mpi::MpiTableBcastImpl::Init(int32_t num_buffers) {
  requests_.resize(num_buffers);
  statuses_.resize(num_buffers);
}

cylon::Status cylon::mpi::MpiTableBcastImpl::BcastBufferSizes(int32_t *buffer,
                                                              int32_t count,
                                                              int32_t bcast_root) const {
  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Bcast(buffer, count, MPI_INT32_T, bcast_root, comm_));
  return Status::OK();
}

cylon::Status cylon::mpi::MpiTableBcastImpl::BcastBufferData(uint8_t *buf_data,
                                                             int32_t send_count,
                                                             int32_t bcast_root) const {
  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Bcast(buf_data,
                                              send_count,
                                              MPI_UINT8_T,
                                              bcast_root,
                                              comm_));
  return Status::OK();
}

cylon::Status cylon::mpi::MpiTableBcastImpl::IbcastBufferData(int32_t buf_idx,
                                                              uint8_t *buf_data,
                                                              int32_t send_count,
                                                              int32_t bcast_root) {
  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Ibcast(buf_data,
                                               send_count,
                                               MPI_UINT8_T,
                                               bcast_root,
                                               comm_,
                                               &requests_[buf_idx]));
  return Status::OK();
}

cylon::Status cylon::mpi::MpiTableBcastImpl::WaitAll(int32_t num_buffers) {
  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Waitall(num_buffers,
                                                requests_.data(),
                                                statuses_.data()));
  return Status::OK();
}

cylon::Status cylon::mpi::Bcast(const std::shared_ptr<cylon::TableSerializer> &serializer,
                                int bcast_root,
                                const std::shared_ptr<cylon::Allocator> &allocator,
                                std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                                std::vector<int32_t> &data_types,
                                const std::shared_ptr<cylon::CylonContext> &ctx) {
  auto comm = GetMpiComm(ctx);
  MpiTableBcastImpl impl(comm);
  return impl.Execute(serializer, allocator, ctx->GetRank(), bcast_root, &received_buffers,
                      &data_types);
}

cylon::Status cylon::mpi::BcastArrowBuffer(std::shared_ptr<arrow::Buffer> &buf,
                                           int bcast_root,
                                           const std::shared_ptr<cylon::CylonContext> &ctx) {
  auto comm = GetMpiComm(ctx);

  // first broadcast the buffer size
  int32_t buf_size = 0;
  if (AmIRoot(bcast_root, ctx)) {
    buf_size = static_cast<int32_t>(buf->size());
  }

  int status = MPI_Bcast(&buf_size, 1, MPI_INT32_T, bcast_root, comm);
  if (status != MPI_SUCCESS) {
    return cylon::Status(cylon::Code::ExecutionError, "MPI_Bcast failed for buf_size broadcast!");
  }

  // allocate arrow Buffers if not the root
  if (!AmIRoot(bcast_root, ctx)) {
    CYLON_ASSIGN_OR_RAISE(buf, arrow::AllocateBuffer(buf_size));
  }

  status = MPI_Bcast((void *) buf->data(), buf_size, MPI_UINT8_T, bcast_root, comm);
  if (status != MPI_SUCCESS) {
    return cylon::Status(cylon::Code::ExecutionError, "MPI_Bcast failed for arrow::Buffer broadcast!");
  }

  return cylon::Status::OK();
}
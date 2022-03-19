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
#include <numeric>
#include <arrow/result.h>

#include "cylon/util/macros.hpp"
#include "cylon/net/mpi/mpi_communicator.hpp"
#include "cylon/net/mpi/mpi_operations.hpp"
#include "cylon/net/ops/base_ops.hpp"
#include "cylon/net/utils.hpp"

cylon::Status cylon::mpi::MpiTableGatherImpl::GatherBufferSizes(const int32_t *send_data,
                                                                int num_buffers,
                                                                int32_t *rcv_data,
                                                                int gather_root) const {
  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Gather(send_data,
                                               num_buffers,
                                               MPI_INT32_T,
                                               rcv_data,
                                               num_buffers,
                                               MPI_INT32_T,
                                               gather_root,
                                               comm_));
  return cylon::Status::OK();
}

cylon::Status cylon::mpi::MpiTableGatherImpl::IgatherBufferData(int buf_idx,
                                                                const uint8_t *send_data,
                                                                int32_t send_count,
                                                                uint8_t *recv_data,
                                                                const std::vector<int32_t> &recv_count,
                                                                const std::vector<int32_t> &displacements,
                                                                int gather_root) {
  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Igatherv(send_data,
                                                 send_count,
                                                 MPI_UINT8_T,
                                                 recv_data,
                                                 recv_count.data(),
                                                 displacements.data(),
                                                 MPI_UINT8_T,
                                                 gather_root,
                                                 comm_,
                                                 &requests_[buf_idx]));
  return cylon::Status::OK();
}

cylon::Status cylon::mpi::MpiTableGatherImpl::WaitAll(int num_buffers) {
  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Waitall(num_buffers,
                                                requests_.data(),
                                                statuses_.data()));
  return cylon::Status::OK();
}

void cylon::mpi::MpiTableGatherImpl::Init(int num_buffers) {
  requests_.resize(num_buffers);
  statuses_.resize(num_buffers);
}

cylon::Status cylon::mpi::Gather(const std::shared_ptr<cylon::TableSerializer> &serializer,
                                 int gather_root,
                                 bool gather_from_root,
                                 const std::shared_ptr<cylon::Allocator> &allocator,
                                 std::vector<int32_t> &all_buffer_sizes,
                                 std::vector<std::shared_ptr<cylon::Buffer>> &receive_buffers,
                                 std::vector<std::vector<int32_t>> &displacements,
                                 const std::shared_ptr<cylon::CylonContext> &ctx) {
  auto comm = GetMpiComm(ctx);
  MpiTableGatherImpl impl(comm);
  return impl.Execute(serializer,
                      allocator,
                      ctx->GetRank(),
                      ctx->GetWorldSize(),
                      gather_root,
                      gather_from_root,
                      &all_buffer_sizes,
                      &receive_buffers,
                      &displacements);
}

cylon::Status cylon::mpi::GatherArrowBuffer(const std::shared_ptr<arrow::Buffer> &buf,
                                            int gather_root,
                                            const std::shared_ptr<cylon::CylonContext> &ctx,
                                            std::vector<std::shared_ptr<arrow::Buffer>> &buffers) {
  auto comm = GetMpiComm(ctx);

  std::vector<int32_t> all_buffer_sizes;
  if (AmIRoot(gather_root, ctx)) {
    all_buffer_sizes.resize(ctx->GetWorldSize(), 0);
  }

  int32_t size = static_cast<int32_t>(buf->size());
  int status = MPI_Gather(&size,
                          1,
                          MPI_INT32_T,
                          all_buffer_sizes.data(),
                          1,
                          MPI_INT32_T,
                          gather_root,
                          comm);
  if (status != MPI_SUCCESS) {
    return cylon::Status(cylon::Code::ExecutionError, "MPI_Gather failed when receiving buffer sizes!");
  }

  CYLON_ASSIGN_OR_RAISE(std::shared_ptr<arrow::Buffer> all_buf, arrow::AllocateBuffer(0));
  std::vector<int32_t> disps;

  if (AmIRoot(gather_root, ctx)) {
    auto total_size = std::accumulate(all_buffer_sizes.begin(), all_buffer_sizes.end(), 0);
    CYLON_ASSIGN_OR_RAISE(all_buf, arrow::AllocateBuffer(total_size));

    disps.resize(ctx->GetWorldSize(), 0);
    std::partial_sum(all_buffer_sizes.begin(), all_buffer_sizes.end() - 1, disps.begin() + 1);
  }

  status = MPI_Gatherv(buf->data(),
                       size,
                       MPI_UINT8_T,
                       (void *) all_buf->data(),
                       all_buffer_sizes.data(),
                       disps.data(),
                       MPI_UINT8_T,
                       gather_root,
                       comm);
  if (status != MPI_SUCCESS) {
    return cylon::Status(cylon::Code::ExecutionError, "MPI_Gatherv failed when receiving buffers!");
  }

  if (gather_root == ctx->GetRank()) {
    buffers.resize(ctx->GetWorldSize());
    for (int i = 0; i < ctx->GetWorldSize(); ++i) {
      buffers[i] = arrow::SliceBuffer(all_buf, disps[i], all_buffer_sizes[i]);
    }
  }

  return cylon::
  Status::OK();
}

cylon::Status cylon::mpi::MpiTableAllgatherImpl::AllgatherBufferSizes(const int32_t *send_data,
                                                                      int num_buffers,
                                                                      int32_t *rcv_data) const {
  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Allgather(send_data,
                                                  num_buffers,
                                                  MPI_INT32_T,
                                                  rcv_data,
                                                  num_buffers,
                                                  MPI_INT32_T,
                                                  comm_));
  return cylon::Status::OK();
}

cylon::Status cylon::mpi::MpiTableAllgatherImpl::IallgatherBufferData(int buf_idx,
                                                                      const uint8_t *send_data,
                                                                      int32_t send_count,
                                                                      uint8_t *recv_data,
                                                                      const std::vector<int32_t> &recv_count,
                                                                      const std::vector<int32_t> &displacements) {
  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Iallgatherv(send_data,
                                                    send_count,
                                                    MPI_UINT8_T,
                                                    recv_data,
                                                    recv_count.data(),
                                                    displacements.data(),
                                                    MPI_UINT8_T,
                                                    comm_,
                                                    &requests_[buf_idx]));
  return cylon::Status::OK();
}

cylon::Status cylon::mpi::MpiTableAllgatherImpl::WaitAll(int num_buffers) {
  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Waitall(num_buffers,
                                                requests_.data(),
                                                statuses_.data()));
  return cylon::Status::OK();
}

cylon::mpi::MpiTableAllgatherImpl::MpiTableAllgatherImpl(MPI_Comm comm)
    : TableAllgatherImpl(), comm_(comm), requests_({}), statuses_({}) {}

void cylon::mpi::MpiTableAllgatherImpl::Init(int num_buffers) {
  requests_.resize(num_buffers);
  statuses_.resize(num_buffers);
}

cylon::Status cylon::mpi::AllGather(const std::shared_ptr<cylon::TableSerializer> &serializer,
                                    const std::shared_ptr<cylon::Allocator> &allocator,
                                    std::vector<int32_t> &all_buffer_sizes,
                                    std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                                    std::vector<std::vector<int32_t>> &displacements,
                                    const std::shared_ptr<cylon::CylonContext> &ctx) {
  auto comm = GetMpiComm(ctx);
  MpiTableAllgatherImpl impl(comm);
  return impl.Execute(serializer, allocator, ctx->GetWorldSize(),
                      &all_buffer_sizes, &received_buffers, &displacements);
}

cylon::Status cylon::mpi::AllGatherArrowBuffer(const std::shared_ptr<arrow::Buffer> &buf,
                                               const std::shared_ptr<cylon::CylonContext> &ctx,
                                               std::vector<std::shared_ptr<arrow::Buffer>> &buffers) {
  auto comm = GetMpiComm(ctx);

  std::vector<int32_t> all_buffer_sizes(ctx->GetWorldSize(), 0);
  int32_t size = static_cast<int32_t>(buf->size());

  int status = MPI_Allgather(&size,
                             1,
                             MPI_INT32_T,
                             all_buffer_sizes.data(),
                             1,
                             MPI_INT32_T,
                             comm);
  if (status != MPI_SUCCESS) {
    return cylon::Status(cylon::Code::ExecutionError, "MPI_Allgather failed when receiving buffer sizes!");
  }

  auto total_size = std::accumulate(all_buffer_sizes.begin(), all_buffer_sizes.end(), 0);
  CYLON_ASSIGN_OR_RAISE(std::shared_ptr<arrow::Buffer> all_buf, arrow::AllocateBuffer(total_size));

  std::vector<int32_t> disps(ctx->GetWorldSize(), 0);
  std::partial_sum(all_buffer_sizes.begin(), all_buffer_sizes.end() - 1, disps.begin() + 1);

  status = MPI_Allgatherv(buf->data(),
                          size,
                          MPI_UINT8_T,
                          (void *) all_buf->data(),
                          all_buffer_sizes.data(),
                          disps.data(),
                          MPI_UINT8_T,
                          comm);
  if (status != MPI_SUCCESS) {
    return cylon::Status(cylon::Code::ExecutionError,
                         "MPI_Allgatherv failed when receiving buffers!");
  }

  buffers.resize(ctx->GetWorldSize());
  for (int i = 0; i < ctx->GetWorldSize(); ++i) {
    buffers[i] = arrow::SliceBuffer(all_buf, disps[i], all_buffer_sizes[i]);
  }

  return cylon::Status::OK();
}

MPI_Comm cylon::mpi::GetMpiComm(const std::shared_ptr<CylonContext> &ctx) {
  return std::static_pointer_cast<cylon::net::MPICommunicator>(ctx->GetCommunicator())->mpi_comm();
}

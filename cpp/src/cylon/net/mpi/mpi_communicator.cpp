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

#include <memory>

#include <arrow/ipc/api.h>

#include "cylon/net/communicator.hpp"
#include "cylon/net/mpi/mpi_communicator.hpp"
#include "cylon/net/mpi/mpi_channel.hpp"
#include "cylon/net/mpi/mpi_operations.hpp"
#include "cylon/scalar.hpp"
#include "cylon/util/macros.hpp"

namespace cylon {
namespace net {

// configs
CommType MPIConfig::Type() {
  return CommType::MPI;
}

std::shared_ptr<MPIConfig> MPIConfig::Make(MPI_Comm comm) {
  return std::make_shared<MPIConfig>(comm);
}

MPIConfig::MPIConfig(MPI_Comm comm) : comm_(comm) {}

MPI_Comm MPIConfig::GetMPIComm() const {
  return comm_;
}

MPIConfig::~MPIConfig() = default;

std::unique_ptr<Channel> MPICommunicator::CreateChannel() const {
  return std::make_unique<MPIChannel>(mpi_comm_);
}

int MPICommunicator::GetRank() const {
  return this->rank;
}
int MPICommunicator::GetWorldSize() const {
  return this->world_size;
}
Status MPICommunicator::Init(const std::shared_ptr<CommConfig> &config) {
  // check if MPI is initialized
  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Initialized(&mpi_initialized_externally));
  mpi_comm_ = std::static_pointer_cast<MPIConfig>(config)->GetMPIComm();

  if (mpi_comm_ && !mpi_initialized_externally) {
    return {Code::Invalid, "non-nullptr MPI_Comm passed without initializing MPI"};
  }

  if (!mpi_initialized_externally) { // if not initialized, init MPI
    RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Init(nullptr, nullptr));
  }

  if (!mpi_comm_) { // set comm_ to world
    mpi_comm_ = MPI_COMM_WORLD;
  }

  // setting errors to return
  MPI_Comm_set_errhandler(mpi_comm_, MPI_ERRORS_RETURN);

  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Comm_rank(mpi_comm_, &this->rank));
  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Comm_size(mpi_comm_, &this->world_size));

  if (rank < 0 || world_size < 0 || rank >= world_size) {
    return {Code::ExecutionError, "Malformed rank :" + std::to_string(rank)
        + " or world size:" + std::to_string(world_size)};
  }

  return Status::OK();
}

void MPICommunicator::Finalize() {
  // finalize only if we initialized MPI
  if (!mpi_initialized_externally) {
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
  }
}
void MPICommunicator::Barrier() {
  MPI_Barrier(mpi_comm_);
}

CommType MPICommunicator::GetCommType() const {
  return MPI;
}

Status MPICommunicator::AllGather(const std::shared_ptr<Table> &table,
                                  std::vector<std::shared_ptr<Table>> *out) const {
  mpi::MpiTableAllgatherImpl impl(mpi_comm_);
  return impl.Execute(table, out);
}

Status MPICommunicator::Gather(const std::shared_ptr<Table> &table,
                               int gather_root,
                               bool gather_from_root,
                               std::vector<std::shared_ptr<Table>> *out) const {
  mpi::MpiTableGatherImpl impl(mpi_comm_);
  return impl.Execute(table, gather_root, gather_from_root, out);
}

Status MPICommunicator::Bcast(std::shared_ptr<Table> *table, int bcast_root) const {
  mpi::MpiTableBcastImpl impl(mpi_comm_);
  return impl.Execute(table, bcast_root, *ctx_ptr);
}

MPI_Comm MPICommunicator::mpi_comm() const {
  return mpi_comm_;
}

Status MPICommunicator::AllReduce(const std::shared_ptr<Column> &values,
                                  net::ReduceOp reduce_op,
                                  std::shared_ptr<Column> *output) const {
  mpi::MpiAllReduceImpl impl(mpi_comm_);
  return impl.Execute(values, reduce_op, output, (*ctx_ptr)->GetMemoryPool());
}

Status MPICommunicator::AllReduce(const std::shared_ptr<Scalar> &value, net::ReduceOp reduce_op,
                                  std::shared_ptr<Scalar> *output) const {
  mpi::MpiAllReduceImpl impl(mpi_comm_);
  return impl.Execute(value, reduce_op, output, (*ctx_ptr)->GetMemoryPool());
}

MPICommunicator::MPICommunicator(const std::shared_ptr<CylonContext> *ctx_ptr)
    : Communicator(ctx_ptr) {}

Status MPICommunicator::Allgather(const std::shared_ptr<Column> &values,
                                  std::vector<std::shared_ptr<Column>> *output) const {
  mpi::MpiAllgatherImpl impl(mpi_comm_);
  return impl.Execute(values, (*ctx_ptr)->GetWorldSize(), output, (*ctx_ptr)->GetMemoryPool());
}

Status MPICommunicator::Allgather(const std::shared_ptr<Scalar> &value,
                                  std::shared_ptr<Column> *output) const {
  mpi::MpiAllgatherImpl impl(mpi_comm_);
  return impl.Execute(value, (*ctx_ptr)->GetWorldSize(), output, (*ctx_ptr)->GetMemoryPool());
}

}  // namespace net
}  // namespace cylon

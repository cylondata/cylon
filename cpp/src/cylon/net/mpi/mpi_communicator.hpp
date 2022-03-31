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

#ifndef CYLON_SRC_CYLON_COMM_MPICOMMUNICATOR_H_
#define CYLON_SRC_CYLON_COMM_MPICOMMUNICATOR_H_

#include <mpi.h>

#include <cylon/net/comm_config.hpp>
#include <cylon/net/communicator.hpp>

namespace cylon {
namespace net {

class MPIConfig : public CommConfig {
 public:
  explicit MPIConfig(MPI_Comm comm = nullptr);

  CommType Type() override;

  ~MPIConfig() override;

  MPI_Comm GetMPIComm() const;

  static std::shared_ptr<MPIConfig> Make(MPI_Comm comm = nullptr);

 private:
  MPI_Comm comm_;
};

class MPICommunicator : public Communicator {
 public:
  explicit MPICommunicator(const std::shared_ptr<CylonContext> *ctx_ptr);
  ~MPICommunicator() override = default;
  Status Init(const std::shared_ptr<CommConfig> &config) override;
  std::unique_ptr<Channel> CreateChannel() const override;
  int GetRank() const override;
  int GetWorldSize() const override;
  void Finalize() override;
  void Barrier() override;
  CommType GetCommType() const override;

  Status AllGather(const std::shared_ptr<Table> &table,
                   std::vector<std::shared_ptr<Table>> *out) const override;

  Status Gather(const std::shared_ptr<Table> &table, int gather_root,
                bool gather_from_root, std::vector<std::shared_ptr<Table>> *out) const override;

  Status Bcast(std::shared_ptr<Table> *table, int bcast_root) const override;

  Status AllReduce(const std::shared_ptr<Column> &values,
                   net::ReduceOp reduce_op,
                   std::shared_ptr<Column> *output) const override;
  Status AllReduce(const std::shared_ptr<Scalar> &value,
                   net::ReduceOp reduce_op,
                   std::shared_ptr<Scalar> *output) const override;

  Status Allgather(const std::shared_ptr<Column> &values,
                   std::vector<std::shared_ptr<Column>> *output) const override;
  Status Allgather(const std::shared_ptr<Scalar> &value,
                   std::shared_ptr<Column> *output) const override;

  MPI_Comm mpi_comm() const;

 private:
  MPI_Comm mpi_comm_ = nullptr;
  int mpi_initialized_externally = 0;
};

}
}
#endif //CYLON_SRC_CYLON_COMM_MPICOMMUNICATOR_H_

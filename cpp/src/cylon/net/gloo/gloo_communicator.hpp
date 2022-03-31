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

#ifndef CYLON_CPP_SRC_CYLON_NET_GLOO_GLOO_COMMUNICATOR_HPP_
#define CYLON_CPP_SRC_CYLON_NET_GLOO_GLOO_COMMUNICATOR_HPP_

#ifdef GLOO_USE_MPI
#include <mpi.h>
#include <gloo/mpi/context.h>
#endif //GLOO_USE_MPI

#include <gloo/rendezvous/store.h>
#include <gloo/rendezvous/context.h>

#include <sys/socket.h>

#include "cylon/net/comm_config.hpp"
#include "cylon/net/communicator.hpp"

namespace cylon {
namespace net {

class GlooConfig : public CommConfig {
 public:
  int rank = 0;
  int world_size = 1;
  bool use_mpi = false;

#ifdef GLOO_USE_MPI
  /*
   * Create a MpiGlooConfig.
   * Pass an MPI communicator to bootstrap Gloo context using MPI (NOTE: Gloo needs to be built
   * with -DUSE_MPI=1 flag, to use MPI communicator).
   */
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
#endif //GLOO_USE_MPI

  // tcp attr
  std::string tcp_hostname = "localhost";
  std::string tcp_iface;
  int tcp_ai_family = AF_UNSPEC;

  // file store configs
  std::string file_store_path;
  std::string store_prefix;

  CommType Type() override;

#ifdef GLOO_USE_MPI
  static std::shared_ptr<GlooConfig> MakeWithMpi(MPI_Comm comm = nullptr);
#endif //GLOO_USE_MPI
};

class GlooCommunicator : public Communicator {
 public:
  explicit GlooCommunicator(const std::shared_ptr<CylonContext> *ctx_ptr);
  Status Init(const std::shared_ptr<CommConfig> &config) override;
  std::unique_ptr<Channel> CreateChannel() const override;
  int GetRank() const override;
  int GetWorldSize() const override;
  void Finalize() override;
  void Barrier() override;
  CommType GetCommType() const override;
  Status AllGather(const std::shared_ptr<Table> &table,
                   std::vector<std::shared_ptr<Table>> *out) const override;
  Status Gather(const std::shared_ptr<Table> &table,
                int gather_root,
                bool gather_from_root,
                std::vector<std::shared_ptr<Table>> *out) const override;
  Status Bcast(std::shared_ptr<Table> *table, int bcast_root) const override;
  Status AllReduce(const std::shared_ptr<Column> &values,
                   net::ReduceOp reduce_op,
                   std::shared_ptr<Column> *output) const override;
  Status AllReduce(const std::shared_ptr<Scalar> &value,
                   net::ReduceOp reduce_op,
                   std::shared_ptr<Scalar> *output) const override;
  Status Allgather(const std::shared_ptr<Column> &values,
                   std::vector<std::shared_ptr<Column>> *output) const override;

 private:
  std::shared_ptr<gloo::Context> gloo_ctx_ = nullptr;
  std::shared_ptr<gloo::transport::Device> dev_ = nullptr;
  std::shared_ptr<gloo::rendezvous::Store> store_ = nullptr;
  std::shared_ptr<gloo::rendezvous::Store> prefix_store_ = nullptr;
};

}
}

#endif //CYLON_CPP_SRC_CYLON_NET_GLOO_GLOO_COMMUNICATOR_HPP_

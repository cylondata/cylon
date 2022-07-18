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

class GlooCommunicator;

class GlooConfig : public CommConfig {
 public:
  explicit GlooConfig(int rank = 0, int world_size = 1, bool use_mpi = false);

#ifdef GLOO_USE_MPI
  explicit GlooConfig(MPI_Comm mpi_comm = MPI_COMM_WORLD);
#endif //GLOO_USE_MPI

  int rank() const;
  int world_size() const;

  void SetTcpHostname(const std::string &tcp_hostname);
  void SetTcpIface(const std::string &tcp_iface);
  void SetTcpAiFamily(int tcp_ai_family);
  void SetFileStorePath(const std::string &file_store_path);
  void SetStorePrefix(const std::string &store_prefix);

  CommType Type() override;

#ifdef GLOO_USE_MPI
  static std::shared_ptr<GlooConfig> MakeWithMpi(MPI_Comm comm = MPI_COMM_NULL);
#endif //GLOO_USE_MPI

  static std::shared_ptr<GlooConfig> Make(int rank, int world_size);

 private:
  friend GlooCommunicator;
  int rank_;
  int world_size_;
  bool use_mpi_;

#ifdef GLOO_USE_MPI
  /*
   * Create a MpiGlooConfig.
   * Pass an MPI communicator to bootstrap Gloo context using MPI (NOTE: Gloo needs to be built
   * with -DUSE_MPI=1 flag, to use MPI communicator).
   */
  MPI_Comm mpi_comm_ = MPI_COMM_WORLD;
#endif //GLOO_USE_MPI

  // tcp attr
  std::string tcp_hostname_;
  std::string tcp_iface_;
  int tcp_ai_family_ = AF_UNSPEC;

  // file store configs
  std::string file_store_path_;
  std::string store_prefix_;
};

class GlooCommunicator : public Communicator {
 public:
  GlooCommunicator(MemoryPool *pool, std::shared_ptr<gloo::Context> gloo_ctx);
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
  Status Bcast(std::shared_ptr<Table> *table,
               int bcast_root,
               const std::shared_ptr<CylonContext> &ctx) const override;
  Status AllReduce(const std::shared_ptr<Column> &values,
                   net::ReduceOp reduce_op,
                   std::shared_ptr<Column> *output) const override;
  Status AllReduce(const std::shared_ptr<Scalar> &value,
                   net::ReduceOp reduce_op,
                   std::shared_ptr<Scalar> *output) const override;
  Status Allgather(const std::shared_ptr<Scalar> &value,
                   std::shared_ptr<Column> *output) const override;
  Status Allgather(const std::shared_ptr<Column> &values,
                   std::vector<std::shared_ptr<Column>> *output) const override;

  static Status Make(const std::shared_ptr<CommConfig> &config,
                     MemoryPool *pool, std::shared_ptr<Communicator> *out);

 private:
  std::shared_ptr<gloo::Context> gloo_ctx_ = nullptr;
};

}
}

#endif //CYLON_CPP_SRC_CYLON_NET_GLOO_GLOO_COMMUNICATOR_HPP_

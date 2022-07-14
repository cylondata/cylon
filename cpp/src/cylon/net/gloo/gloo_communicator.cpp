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

#include <gloo/transport/tcp/attr.h>
#include <gloo/transport/tcp/device.h>
#include <gloo/rendezvous/file_store.h>
#include <gloo/rendezvous/prefix_store.h>
#include <gloo/barrier.h>


#include "gloo_communicator.hpp"
#include "cylon/util/macros.hpp"
#include "cylon/net/serialize.hpp"
#include "cylon/serialize/table_serialize.hpp"
#include "cylon/net/ops/base_ops.hpp"
#include "cylon/net/gloo/gloo_operations.hpp"
#include "gloo_channel.hpp"

namespace cylon {
namespace net {

Status GlooCommunicator::Make(const std::shared_ptr<CommConfig> &config,
                              MemoryPool *pool,
                              std::shared_ptr<Communicator> *out) {
  const auto &gloo_config = std::static_pointer_cast<GlooConfig>(config);

  gloo::transport::tcp::attr attr;
  attr.hostname = gloo_config->tcp_hostname_;
  attr.iface = gloo_config->tcp_iface_;
  attr.ai_family = gloo_config->tcp_ai_family_;

  // create device
  auto dev = gloo::transport::tcp::CreateDevice(attr);

  std::shared_ptr<gloo::Context> gloo_ctx;
  if (gloo_config->use_mpi_) {
#ifdef GLOO_USE_MPI
    int res;
    RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Initialized(&res));
    if (res) {
      if (gloo_config->mpi_comm_ == MPI_COMM_NULL) {
        gloo_ctx = std::make_shared<gloo::mpi::Context>(MPI_COMM_WORLD);
      } else {
        gloo_ctx = std::make_shared<gloo::mpi::Context>(gloo_config->mpi_comm_);
      }
    } else { // MPI is not initialized. Ask gloo to initialize MPI
      gloo_ctx = gloo::mpi::Context::createManaged();
    }
    ((gloo::mpi::Context &) *gloo_ctx).connectFullMesh(dev);
#else
    return {Code::Invalid, "Gloo does not contain mpi headers!"};
#endif // GLOO_USE_MPI
  } else {
    // store and prefix store
    auto store = std::make_shared<gloo::rendezvous::FileStore>(gloo_config->file_store_path_);
    auto prefix_store = std::make_shared<gloo::rendezvous::PrefixStore>(gloo_config->store_prefix_,
                                                                        *store);

    gloo_ctx = std::make_shared<gloo::rendezvous::Context>(gloo_config->rank_,
                                                           gloo_config->world_size_);
    ((gloo::rendezvous::Context &) *gloo_ctx).connectFullMesh(*prefix_store, dev);

  }
  *out = std::make_shared<GlooCommunicator>(pool, std::move(gloo_ctx));
  return Status::OK();
}

std::unique_ptr<Channel> GlooCommunicator::CreateChannel() const {
  return std::make_unique<GlooChannel>(gloo_ctx_.get());
}

int GlooCommunicator::GetRank() const {
  return gloo_ctx_->rank;
}

int GlooCommunicator::GetWorldSize() const {
  return gloo_ctx_->size;
}

void GlooCommunicator::Finalize() {}

void GlooCommunicator::Barrier() {
  gloo::BarrierOptions opts(gloo_ctx_);
  gloo::barrier(opts);
}

CommType GlooCommunicator::GetCommType() const {
  return GLOO;
}

Status GlooCommunicator::AllGather(const std::shared_ptr<Table> &table,
                                   std::vector<std::shared_ptr<Table>> *out) const {
  GlooTableAllgatherImpl impl(&gloo_ctx_);
  return impl.Execute(table, out);
}

Status GlooCommunicator::Gather(const std::shared_ptr<Table> &table,
                                int gather_root,
                                bool gather_from_root,
                                std::vector<std::shared_ptr<Table>> *out) const {
  GlooTableGatherImpl impl(&gloo_ctx_);
  return impl.Execute(table, gather_root, gather_from_root, out);
}

Status GlooCommunicator::Bcast(std::shared_ptr<Table> *table,
                               int bcast_root,
                               const std::shared_ptr<CylonContext> &ctx) const {
  GlooTableBcastImpl impl(&gloo_ctx_);
  return impl.Execute(table, bcast_root, ctx);
}

Status GlooCommunicator::AllReduce(const std::shared_ptr<Column> &values,
                                   net::ReduceOp reduce_op,
                                   std::shared_ptr<Column> *output) const {
  GlooAllReduceImpl impl(&gloo_ctx_);
  return impl.Execute(values, reduce_op, output, pool);
}

Status GlooCommunicator::AllReduce(const std::shared_ptr<Scalar> &value,
                                   net::ReduceOp reduce_op,
                                   std::shared_ptr<Scalar> *output) const {
  GlooAllReduceImpl impl(&gloo_ctx_);
  return impl.Execute(value, reduce_op, output, pool);
}

GlooCommunicator::GlooCommunicator(MemoryPool *pool,
                                   std::shared_ptr<gloo::Context> gloo_ctx)
    : Communicator(pool, gloo_ctx->rank, gloo_ctx->size), gloo_ctx_(std::move(gloo_ctx)) {}

Status GlooCommunicator::Allgather(const std::shared_ptr<Column> &values,
                                   std::vector<std::shared_ptr<Column>> *output) const {
  GlooAllgatherImpl impl(&gloo_ctx_);
  return impl.Execute(values, gloo_ctx_->size, output, pool);
}
Status GlooCommunicator::Allgather(const std::shared_ptr<Scalar> &value,
                                   std::shared_ptr<Column> *output) const {
  GlooAllgatherImpl impl(&gloo_ctx_);
  return impl.Execute(value, gloo_ctx_->size, output, pool);
}

CommType GlooConfig::Type() { return GLOO; }

#ifdef GLOO_USE_MPI
std::shared_ptr<GlooConfig> GlooConfig::MakeWithMpi(MPI_Comm comm) {
  return std::make_shared<GlooConfig>(comm);
}

GlooConfig::GlooConfig(MPI_Comm mpi_comm)
    : rank_(-1), world_size_(-1), use_mpi_(true), mpi_comm_(mpi_comm) {}
#endif //GLOO_USE_MPI

std::shared_ptr<GlooConfig> GlooConfig::Make(int rank, int world_size) {
  return std::make_shared<GlooConfig>(rank, world_size);
}

GlooConfig::GlooConfig(int rank, int world_size, bool use_mpi)
    : rank_(rank), world_size_(world_size), use_mpi_(use_mpi) {}

void GlooConfig::SetTcpHostname(const std::string &tcp_hostname) {
  GlooConfig::tcp_hostname_ = tcp_hostname;
}

void GlooConfig::SetTcpIface(const std::string &tcp_iface) {
  GlooConfig::tcp_iface_ = tcp_iface;
}

void GlooConfig::SetTcpAiFamily(int tcp_ai_family) {
  GlooConfig::tcp_ai_family_ = tcp_ai_family;
}

void GlooConfig::SetFileStorePath(const std::string &file_store_path) {
  GlooConfig::file_store_path_ = file_store_path;
}

void GlooConfig::SetStorePrefix(const std::string &store_prefix) {
  GlooConfig::store_prefix_ = store_prefix;
}
int GlooConfig::rank() const {
  return rank_;
}
int GlooConfig::world_size() const {
  return world_size_;
}

}
}

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

Status GlooCommunicator::Init(const std::shared_ptr<CommConfig> &config) {
  const auto &gloo_config = std::static_pointer_cast<GlooConfig>(config);

  gloo::transport::tcp::attr attr;
  attr.hostname = gloo_config->tcp_hostname;
  attr.iface = gloo_config->tcp_iface;
  attr.ai_family = gloo_config->tcp_ai_family;

  // create device
  dev_ = gloo::transport::tcp::CreateDevice(attr);

  if (gloo_config->use_mpi) {
#ifdef GLOO_USE_MPI
    int res;
    RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Initialized(&res));
    if (res) {
      gloo_ctx_ = std::make_shared<gloo::mpi::Context>(gloo_config->mpi_comm);
    } else { // MPI is not initialized. Ask gloo to initialize MPI
      gloo_ctx_ = gloo::mpi::Context::createManaged();
    }
    ((gloo::mpi::Context &) *gloo_ctx_).connectFullMesh(dev_);

    // update rank and world size
    rank = gloo_ctx_->rank;
    world_size = gloo_ctx_->size;
#else
    return {Code::Invalid, "Gloo does not contain mpi headers!"};
#endif // GLOO_USE_MPI
  } else {
    // store and prefix store
    store_ = std::make_shared<gloo::rendezvous::FileStore>(gloo_config->file_store_path);
    prefix_store_ = std::make_shared<gloo::rendezvous::PrefixStore>(gloo_config->store_prefix,
                                                                    *store_);

    gloo_ctx_ = std::make_shared<gloo::rendezvous::Context>(rank, world_size);
    ((gloo::rendezvous::Context &) *gloo_ctx_).connectFullMesh(*prefix_store_, dev_);

    rank = gloo_config->rank;
    world_size = gloo_config->world_size;
  }
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

Status GlooCommunicator::Bcast(std::shared_ptr<Table> *table, int bcast_root) const {
  GlooTableBcastImpl impl(&gloo_ctx_);
  return impl.Execute(table, bcast_root, *ctx_ptr);
}

Status GlooCommunicator::AllReduce(const std::shared_ptr<Column> &values,
                                   net::ReduceOp reduce_op,
                                   std::shared_ptr<Column> *output) const {
  GlooAllReduceImpl impl(&gloo_ctx_);
  return impl.Execute(values, reduce_op, output, (*ctx_ptr)->GetMemoryPool());
}

Status GlooCommunicator::AllReduce(const std::shared_ptr<Scalar> &value,
                                   net::ReduceOp reduce_op,
                                   std::shared_ptr<Scalar> *output) const {
  GlooAllReduceImpl impl(&gloo_ctx_);
  return impl.Execute(value, reduce_op, output, (*ctx_ptr)->GetMemoryPool());
}

GlooCommunicator::GlooCommunicator(const std::shared_ptr<CylonContext> *ctx_ptr)
    : Communicator(ctx_ptr) {}

Status GlooCommunicator::Allgather(const std::shared_ptr<Column> &values,
                                   std::vector<std::shared_ptr<Column>> *output) const {
  GlooAllgatherImpl impl(&gloo_ctx_);
  return impl.Execute(values, gloo_ctx_->size, output, (*ctx_ptr)->GetMemoryPool());
}

CommType GlooConfig::Type() { return GLOO; }

#ifdef GLOO_USE_MPI
std::shared_ptr<GlooConfig> GlooConfig::MakeWithMpi(MPI_Comm comm) {
  auto config = std::make_shared<GlooConfig>();
  config->use_mpi = true;
  config->mpi_comm = comm;
  return config;
}
#endif //GLOO_USE_MPI
}
}

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

#include "gloo_communicator.hpp"
#include "cylon/util/macros.hpp"

namespace cylon {
namespace net {

Status GlooCommunicator::Init(const std::shared_ptr<CommConfig> &config) {
  const auto &gloo_config = std::static_pointer_cast<GlooConfig>(config);

  rank = gloo_config->rank;
  world_size = gloo_config->world_size;

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
  }
  return Status::OK();
}
std::unique_ptr<Channel> GlooCommunicator::CreateChannel() const {
  return std::unique_ptr<Channel>();
}
int GlooCommunicator::GetRank() const {
  return gloo_ctx_->rank;
}
int GlooCommunicator::GetWorldSize() const {
  return gloo_ctx_->size;
}
void GlooCommunicator::Finalize() {

}
void GlooCommunicator::Barrier() {

}
CommType GlooCommunicator::GetCommType() const {
  return GLOO;
}
Status GlooCommunicator::AllGather(const std::shared_ptr<Table> &table,
                                   std::vector<std::shared_ptr<Table>> *out) const {
  return Status::OK();
}
Status GlooCommunicator::Gather(const std::shared_ptr<Table> &table,
                                int gather_root,
                                bool gather_from_root,
                                std::vector<std::shared_ptr<Table>> *out) const {
  return Status::OK();
}
Status GlooCommunicator::Bcast(std::shared_ptr<Table> *table, int bcast_root) const {
  return Status::OK();
}
Status GlooCommunicator::AllReduce(const std::shared_ptr<Column> &values,
                                   net::ReduceOp reduce_op,
                                   std::shared_ptr<Column> *output) const {
  return Status::OK();
}
Status GlooCommunicator::AllReduce(const std::shared_ptr<Scalar> &value,
                                   net::ReduceOp reduce_op,
                                   std::shared_ptr<Scalar> *output) const {
  return Status::OK();
}

GlooCommunicator::GlooCommunicator(const std::shared_ptr<CylonContext> *ctx_ptr)
    : Communicator(ctx_ptr) {}
CommType GlooConfig::Type() { return GLOO; }
}
}

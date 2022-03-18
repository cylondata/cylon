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
#include <gloo/allgather.h>
#include <gloo/allgatherv.h>

#include "gloo_communicator.hpp"
#include "cylon/util/macros.hpp"
#include "cylon/net/serialize.hpp"
#include "cylon/net/utils.hpp"
#include "cylon/serialize/table_serialize.hpp"
#include "cylon/arrow/arrow_buffer.hpp"
#include "cylon/net/ops/base_ops.hpp"

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
  return std::unique_ptr<Channel>();
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

class GlooTableAllgatherImpl : public TableAllgatherImpl {
 public:
  GlooTableAllgatherImpl(const int num_buffers, const std::shared_ptr<gloo::Context> *ctx_ptr)
      : TableAllgatherImpl(num_buffers), ctx_ptr_(ctx_ptr) {}

  Status AllgatherBufferSizes(const int32_t *send_data, int32_t *rcv_data) override {
    gloo::AllgatherOptions opts(*ctx_ptr_);
    opts.setInput(const_cast<int32_t *>(send_data), num_buffers_);
    opts.setOutput(rcv_data, num_buffers_ * (*ctx_ptr_)->size);

    gloo::allgather(opts);

    return Status::OK();
  }

  // gloo doesn't have non-blocking collectives. So, do blocking call here!
  Status IallgatherBufferData(int buf_idx,
                              const uint8_t *send_data,
                              int32_t send_count,
                              uint8_t *recv_data,
                              const std::vector<int32_t> &recv_count,
                              const std::vector<int32_t> &displacements) override {
    gloo::AllgathervOptions opts(*ctx_ptr_);
    opts.setInput(const_cast<uint8_t *>(send_data), send_count);
    opts.setOutput(recv_data, std::vector<size_t>(recv_count.begin(), recv_count.end()));

    gloo::allgatherv(opts);
    return Status::OK();
  }

  Status WaitAll() override {
    return Status::OK();
  }

 private:
  const std::shared_ptr<gloo::Context> *ctx_ptr_;
};

Status GlooCommunicator::AllGather(const std::shared_ptr<Table> &table,
                                   std::vector<std::shared_ptr<Table>> *out) const {
  std::shared_ptr<TableSerializer> serializer;
  RETURN_CYLON_STATUS_IF_FAILED(CylonTableSerializer::Make(table, &serializer));
  const auto &ctx = table->GetContext();
  auto *pool = ToArrowPool(ctx);

  const auto &allocator = std::make_shared<ArrowAllocator>(pool);
  std::vector<std::shared_ptr<Buffer>> receive_buffers;

  std::vector<int32_t> buffer_sizes_per_table;
  //  |b_0, ..., b_n-1|...|b_0, ..., b_n-1|
  //   <--- tbl_0 --->     <--- tbl_m --->

  std::vector<std::vector<int32_t>> all_disps;
  //  |t_0, ..., t_m-1|...|t_0, ..., t_m-1|
  //   <--- buf_0 --->     <--- buf_n --->

  GlooTableAllgatherImpl impl(serializer->getNumberOfBuffers(), &gloo_ctx_);
  RETURN_CYLON_STATUS_IF_FAILED(impl.Execute(serializer,
                                             allocator,
                                             buffer_sizes_per_table,
                                             receive_buffers,
                                             all_disps,
                                             world_size));


  // need to reshape all_disps for per-table basis
  auto buffer_offsets_per_table = ReshapeDispToPerTable(all_disps);

  const int num_tables = (int) all_disps[0].size();
  return DeserializeTables(ctx, table->get_table()->schema(), num_tables, receive_buffers,
                           buffer_sizes_per_table, buffer_offsets_per_table, out);
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

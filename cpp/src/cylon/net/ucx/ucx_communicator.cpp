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
#include <glog/logging.h>
#include <cylon/net/communicator.hpp>
#include <cylon/net/ucx/ucx_communicator.hpp>
#include <cylon/net/ucx/ucx_channel.hpp>
#include <cylon/util/macros.hpp>
#include <cylon/net/ucc/ucc_operations.hpp>

namespace cylon {
namespace net {
void UCXConfig::DummyConfig(int dummy) {
  this->AddConfig("Dummy", &dummy);
}
int UCXConfig::GetDummyConfig() {
  return *reinterpret_cast<int *>(this->GetConfig("Dummy"));
}

CommType UCXConfig::Type() {
  return CommType::UCX;
}

std::shared_ptr<UCXConfig> UCXConfig::Make() {
  return std::make_shared<UCXConfig>();
}

std::unique_ptr<Channel> UCXCommunicator::CreateChannel() const {
  return std::make_unique<UCXChannel>(this);
}

int UCXCommunicator::GetRank() const {
  return this->rank;
}
int UCXCommunicator::GetWorldSize() const {
  return this->world_size;
}

static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                  void *coll_info, void **req) {
  auto comm = (MPI_Comm)coll_info;
  MPI_Request request;

  MPI_Iallgather(sbuf, (int)msglen, MPI_BYTE, rbuf, (int)msglen, MPI_BYTE, comm,
                 &request);
  *req = (void *)request;
  return UCC_OK;
}

static ucc_status_t oob_allgather_test(void *req) {
  auto request = (MPI_Request)req;
  int completed;

  MPI_Test(&request, &completed, MPI_STATUS_IGNORE);
  return completed ? UCC_OK : UCC_INPROGRESS;
}

static ucc_status_t oob_allgather_free(void *req) { return UCC_OK; }

Status UCXCommunicator::Init(const std::shared_ptr<CommConfig> &config) {
  CYLON_UNUSED(config);
  // Check init functions
  int initialized;
  // Int variable used when iterating
  int sIndx;
  // Address of the UCP Worker for receiving
  cylon::ucx::ucxWorkerAddr *ucpRecvWorkerAddr;
  // Address of the UCP Worker for sending
  cylon::ucx::ucxWorkerAddr *ucpSendWorkerAddr;

  // Status check when creating end-points
  ucs_status_t ucxStatus;
  // Variable to hold the current ucp address
  ucp_address_t *address;

  // MPI init
  MPI_Initialized(&initialized);
  if (!initialized) {
    RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Init(nullptr, nullptr));
  }

  // Get the rank for checking send to self, and initializations
  MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &this->world_size);

  // Init context
  RETURN_CYLON_STATUS_IF_FAILED(cylon::ucx::initContext(&ucpContext, nullptr));

  // Init recv worker and get address
  ucpRecvWorkerAddr = cylon::ucx::initWorker(ucpContext, &ucpRecvWorker);
  // Init send worker
  ucpSendWorkerAddr = cylon::ucx::initWorker(ucpContext, &ucpSendWorker);

  //  Gather all worker addresses
  // All addresses buffer for allGather
  auto allAddresses = std::make_unique<uint8_t[]>(ucpRecvWorkerAddr->addrSize * world_size);
  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Allgather(ucpRecvWorkerAddr->addr,
                                                  (int) ucpRecvWorkerAddr->addrSize,
                                                  MPI_BYTE,
                                                  allAddresses.get(),
                                                  (int) ucpRecvWorkerAddr->addrSize,
                                                  MPI_BYTE,
                                                  MPI_COMM_WORLD));

  // Iterate and set the sends
  for (sIndx = 0; sIndx < this->world_size; sIndx++) {
    ucp_ep_params_t epParams;
    ucp_ep_h ep;

    // If not self, then check if the worker address has been received.
    //  If self,then assign local worker
    if (this->rank != sIndx) {
      address = reinterpret_cast<ucp_address_t *>(allAddresses.get()
          + sIndx * ucpRecvWorkerAddr->addrSize);
    } else {
      address = ucpRecvWorkerAddr->addr;
    }

    // Set params for the endpoint
    epParams.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
        UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    epParams.address = address;
    epParams.err_mode = UCP_ERR_HANDLING_MODE_NONE;

    // Create an endpoint
    ucxStatus = ucp_ep_create(ucpSendWorker, &epParams, &ep);

    endPointMap[sIndx] = ep;
    // Check if the endpoint was created properly
    if (ucxStatus != UCS_OK) {
      LOG(FATAL) << "Error when creating the endpoint.";
      return {Code::ExecutionError,
              "Error when creating the endpoint: " + std::string(ucs_status_string(ucxStatus))};
    }
  }

  // temporary code to initialize UCC team and context
  ucc_context_params_t ctx_params;
  ucc_team_params_t team_params;
  ucc_context_config_h ctx_config;
  ucc_status_t status;
  ucc_lib_h lib;
  ucc_lib_config_h lib_config;

  ucc_lib_params_t lib_params = {.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE,
                                 .thread_mode = UCC_THREAD_SINGLE,
                                 .coll_types = {},
                                 .reduction_types = {},
                                 .sync_type = {}};

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_lib_config_read(nullptr, nullptr, &lib_config));
  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_init(&lib_params, lib_config, &lib));
  ucc_lib_config_release(lib_config);

  ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB;
  ctx_params.oob.allgather = oob_allgather;
  ctx_params.oob.req_test = oob_allgather_test;
  ctx_params.oob.req_free = oob_allgather_free;
  ctx_params.oob.coll_info = (void *)MPI_COMM_WORLD;
  ctx_params.oob.n_oob_eps = static_cast<uint32_t>(this->world_size);
  ctx_params.oob.oob_ep = static_cast<uint32_t>(rank);

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_context_config_read(lib, nullptr, &ctx_config));
  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_context_create(lib, &ctx_params, ctx_config, &uccContext));
  ucc_context_config_release(ctx_config);

  team_params.mask = UCC_TEAM_PARAM_FIELD_OOB;
  team_params.oob.allgather = oob_allgather;
  team_params.oob.req_test = oob_allgather_test;
  team_params.oob.req_free = oob_allgather_free;
  team_params.oob.coll_info = (void *)MPI_COMM_WORLD;
  team_params.oob.n_oob_eps = static_cast<uint32_t>(this->world_size);
  team_params.oob.oob_ep = static_cast<uint32_t>(rank);

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_team_create_post(&uccContext, 1, &team_params, &uccTeam));
  while (UCC_INPROGRESS == (status = ucc_team_create_test(uccTeam))) {
    RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_context_progress(uccContext));
  };

  RETURN_CYLON_STATUS_IF_UCC_FAILED(status);

  // Cleanup
  delete (ucpRecvWorkerAddr);
  delete (ucpSendWorkerAddr);

  return Status::OK();
}
void UCXCommunicator::Finalize() {
  ucp_cleanup(ucpContext);
  MPI_Finalize();
}

void UCXCommunicator::Barrier() {
  MPI_Barrier(MPI_COMM_WORLD);
}

CommType UCXCommunicator::GetCommType() const {
  return UCX;
}

Status UCXCommunicator::AllGather(const std::shared_ptr<Table> &table,
                                  std::vector<std::shared_ptr<Table>> *out) const {
  ucc::UccTableAllgatherImpl impl(uccTeam, uccContext, this->world_size);
  return impl.Execute(table, out);
}

Status UCXCommunicator::Gather(const std::shared_ptr<Table> &table,
                               int gather_root,
                               bool gather_from_root,
                               std::vector<std::shared_ptr<Table>> *out) const {
  CYLON_UNUSED(table);
  CYLON_UNUSED(gather_root);
  CYLON_UNUSED(gather_from_root);
  CYLON_UNUSED(out);
  return {Code::NotImplemented, "All gather not implemented yet for ucx"};
}

Status UCXCommunicator::Bcast(std::shared_ptr<Table> *table, int bcast_root) const {
  CYLON_UNUSED(table);
  CYLON_UNUSED(bcast_root);
  return {Code::NotImplemented, "All gather not implemented yet for ucx"};
}
Status UCXCommunicator::AllReduce(const std::shared_ptr<Column> &column,
                                  net::ReduceOp reduce_op,
                                  std::shared_ptr<Column> *output) const {
  CYLON_UNUSED(column);
  CYLON_UNUSED(reduce_op);
  CYLON_UNUSED(output);
  return {Code::NotImplemented, "Allreduce not implemented yet for ucx"};
}

UCXCommunicator::UCXCommunicator(const std::shared_ptr<CylonContext> *ctx_ptr)
    : Communicator(ctx_ptr) {}
Status UCXCommunicator::AllReduce(const std::shared_ptr<Scalar> &values,
                                  net::ReduceOp reduce_op,
                                  std::shared_ptr<Scalar> *output) const {
  CYLON_UNUSED(values);
  CYLON_UNUSED(reduce_op);
  CYLON_UNUSED(output);
  return {Code::NotImplemented, "Allreduce not implemented yet for ucx"};
}
Status UCXCommunicator::Allgather(const std::shared_ptr<Column> &values,
                                  std::vector<std::shared_ptr<Column>> *output) const {
  CYLON_UNUSED(values);
  CYLON_UNUSED(output);
  return {Code::NotImplemented, "Allgather not implemented yet for ucx"};
}
Status UCXCommunicator::Allgather(const std::shared_ptr<Scalar> &value,
                                  std::shared_ptr<Column> *output) const {
  CYLON_UNUSED(value);
  CYLON_UNUSED(output);
  return {Code::NotImplemented, "Allgather not implemented yet for ucx"};
}
}  // namespace net
}  // namespace cylon

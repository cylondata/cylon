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
    MPI_Init(nullptr, nullptr);
  }

  // Get the rank for checking send to self, and initializations
  MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &this->world_size);

  // Init context
  initialized = cylon::ucx::initContext(&ucpContext, nullptr);
  if (initialized != 0) {
    LOG(FATAL) << "Error occurred when creating UCX context";
  }

  // Init recv worker and get address
  ucpRecvWorkerAddr = cylon::ucx::initWorker(ucpContext, &ucpRecvWorker);
  // Init send worker
  ucpSendWorkerAddr = cylon::ucx::initWorker(ucpContext, &ucpSendWorker);

  //  Gather all worker addresses
  // All addresses buffer for allGather
  auto allAddresses = std::make_unique<uint8_t[]>(ucpRecvWorkerAddr->addrSize * world_size);
  MPI_Allgather(ucpRecvWorkerAddr->addr,
                (int) ucpRecvWorkerAddr->addrSize,
                MPI_BYTE,
                allAddresses.get(),
                (int) ucpRecvWorkerAddr->addrSize,
                MPI_BYTE,
                MPI_COMM_WORLD);

  // Iterate and set the sends
  for (sIndx = 0; sIndx < this->world_size; sIndx++) {
    ucp_ep_params_t epParams;
    ucp_ep_h ep;

    // If not self, then check if the worker address has been received.
    //  If self,then assign local worker
    if (this->rank != sIndx) {
      address = reinterpret_cast<ucp_address_t *>(allAddresses.get() + sIndx * ucpRecvWorkerAddr->addrSize);
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
      return Status(ucxStatus, "This is an error from UCX");
    }
  }

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

Status UCXSyncCommunicator::AllGather(const std::shared_ptr<Table> &table,
                                      std::vector<std::shared_ptr<Table>> *out) const {
  CYLON_UNUSED(table);
  CYLON_UNUSED(out);
  return {Code::NotImplemented, "All gather not implemented yet for ucx"};
}

Status UCXSyncCommunicator::Gather(const std::shared_ptr<Table> &table,
                                   int gather_root,
                                   bool gather_from_root,
                                   std::vector<std::shared_ptr<Table>> *out) const {
  CYLON_UNUSED(table);
  CYLON_UNUSED(gather_root);
  CYLON_UNUSED(gather_from_root);
  CYLON_UNUSED(out);
  return {Code::NotImplemented, "All gather not implemented yet for ucx"};
}

Status UCXSyncCommunicator::Bcast(const std::shared_ptr<cylon::CylonContext> &ctx,
                                  const std::shared_ptr<Table> &table,
                                  int bcast_root,
                                  std::shared_ptr<Table> *out) const {
  CYLON_UNUSED(table);
  CYLON_UNUSED(bcast_root);
  CYLON_UNUSED(out);
  return {Code::NotImplemented, "All gather not implemented yet for ucx"};
}
}  // namespace net
}  // namespace cylon

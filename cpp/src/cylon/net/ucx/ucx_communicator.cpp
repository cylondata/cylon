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

#include "net/communicator.hpp"
#include "ucx_communicator.hpp"
#include "ucx_channel.hpp"

namespace cylon {
namespace net {
// TODO Sandeepa put config and context init here?
void UCXConfig::DummyConfig(int dummy) {
  this->AddConfig("Dummy", &dummy);
}
int UCXConfig::GetDummyConfig() {
  return *reinterpret_cast<int *>(this->GetConfig("Dummy"));
}

CommType UCXConfig::Type() {
  return CommType::UCX;
}
// TODO Sandeepa add the config to the list?
std::shared_ptr<UCXConfig> UCXConfig::Make() {
  return std::make_shared<UCXConfig>();
}

Channel *UCXCommunicator::CreateChannel() {
  auto newChannel = new UCXChannel();
  newChannel->linkCommunicator(this);
  return newChannel;
}

int UCXCommunicator::GetRank() {
  return this->rank;
}
int UCXCommunicator::GetWorldSize() {
  return this->world_size;
}
void UCXCommunicator::Init(const std::shared_ptr<CommConfig> &config) {
  // Check init functions
  int initialized;
  // Int variable used when iterating
  int sIndx;
  // UCP Context - Holds a UCP communication instance's global information.
  ucp_context_h ucpContext;
  // Address of the UCP Worker for receiving
  cylon::ucx::ucxWorkerAddr *ucpRecvWorkerAddr;
  // TODO Sandeepa check to see if you could pass nullptr
  // Address of the UCP Worker for sending
  cylon::ucx::ucxWorkerAddr *ucpSendWorkerAddr;
  // All addresses buffer for allGather
  ucp_address_t * allAddresses;
  // Status check when creating end-points
  ucs_status_t ucxStatus;
  // Variable to hold the current ucp address
  ucp_address_t * address;

  // MPI init
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(nullptr, nullptr);
  }

  // Get the rank for checking send to self, and initializations
  MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
  // TODO Sandeepa can you go with the assumption that ranks don't change?
  MPI_Comm_size(MPI_COMM_WORLD, &this->world_size);

  // Init context
  initialized = cylon::ucx::initContext(&ucpContext, nullptr);
  if (initialized != 0) {
    LOG(FATAL) << "Error occurred when creating UCX context";
  }

//  // Init recv worker and get address
//  cylon::ucx::initWorker(ucpContext,
//                         &ucpRecvWorker,
//                         ucpRecvWorkerAddr);
//  // Init send worker
//  cylon::ucx::initWorker(ucpContext,
//                         &ucpSendWorker,
//                         ucpSendWorkerAddr);

  // Init recv worker and get address
  ucpRecvWorkerAddr = cylon::ucx::initWorker(ucpContext,
                         &ucpRecvWorker);
  // Init send worker
  ucpSendWorkerAddr = cylon::ucx::initWorker(ucpContext,
                         &ucpSendWorker);

  allAddresses = (ucp_address_t *)malloc(ucpRecvWorkerAddr->addrSize*world_size);
  MPI_Allgather(ucpRecvWorkerAddr->addr,
                (int)ucpRecvWorkerAddr->addrSize,
                MPI_BYTE,
                allAddresses,
                (int)ucpRecvWorkerAddr->addrSize,
                MPI_BYTE,
                MPI_COMM_WORLD);

  // Iterate and set the sends
  for (sIndx = 0; sIndx < this->world_size; sIndx++) {
    ucp_ep_params_t epParams;
    ucp_ep_h ep;

    // If not self, then check if the worker address has been received.
    //  If self,then assign local worker
    if (this->rank != sIndx) {
      char *p = (char*)allAddresses;
      p += sIndx*ucpRecvWorkerAddr->addrSize;
      address = (ucp_address_t*)p;
    } else {
      address = ucpRecvWorkerAddr->addr;
    }

    // Set params for the endpoint
    epParams.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
        UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    epParams.address = address;
    epParams.err_mode = UCP_ERR_HANDLING_MODE_NONE;

    // Create an endpoint
    ucxStatus = ucp_ep_create(ucpSendWorker,
                              &epParams,
                              &ep);

    endPointMap[sIndx] = ep;
    // Check if the endpoint was created properly
    if (ucxStatus != UCS_OK) {
      LOG(FATAL)
          << "Error when creating the endpoint.";
    }
  }

  // TODO Add all for cleaning
  //  Need to add *?
  // Cleanup
  delete(ucpRecvWorkerAddr);
  delete(ucpSendWorkerAddr);
}
void UCXCommunicator::Finalize() {
  // TODO Sandeepa get UCP context somewhere
//  ucp_context_h dummy_context;
//  ucp_cleanup(dummy_context);
  MPI_Finalize();
}
void UCXCommunicator::Barrier() {
  MPI_Barrier(MPI_COMM_WORLD);
}

CommType UCXCommunicator::GetCommType() {
  return UCX;
}
}  // namespace net
}  // namespace cylon

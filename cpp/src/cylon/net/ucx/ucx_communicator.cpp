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
//#include <ucp/api/ucp.h>

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
  return new UCXChannel();
}

int UCXCommunicator::GetRank() {
  return this->rank;
}
int UCXCommunicator::GetWorldSize() {
  return this->world_size;
}
void UCXCommunicator::Init(const std::shared_ptr<CommConfig> &config) {
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(nullptr, nullptr);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &this->world_size);
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

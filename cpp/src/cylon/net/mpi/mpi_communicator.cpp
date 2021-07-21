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

#include <mpi.h>
#include <memory>

#include <cylon/net/communicator.hpp>
#include <cylon/net/mpi/mpi_communicator.hpp>
#include <cylon/net/mpi/mpi_channel.hpp>
#include <cylon/net/mpi/mpi_operations.hpp>

namespace cylon {
namespace net {
// configs
void MPIConfig::DummyConfig(int dummy) {
  this->AddConfig("Dummy", &dummy);
}
int MPIConfig::GetDummyConfig() {
  return *reinterpret_cast<int *>(this->GetConfig("Dummy"));
}

CommType MPIConfig::Type() {
  return CommType::MPI;
}
std::shared_ptr<MPIConfig> MPIConfig::Make() {
  return std::make_shared<MPIConfig>();
}

MPIConfig::~MPIConfig() {}

Channel *MPICommunicator::CreateChannel() {
  return new MPIChannel();
}

int MPICommunicator::GetRank() {
  return this->rank;
}
int MPICommunicator::GetWorldSize() {
  return this->world_size;
}
Status MPICommunicator::Init(const std::shared_ptr<CommConfig> &config) {
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(nullptr, nullptr);
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &this->world_size);

  return Status::OK();
}
void MPICommunicator::Finalize() {
  int finalized;
  MPI_Finalized(&finalized);
  if (!finalized) {
	MPI_Finalize();
  } 
}
void MPICommunicator::Barrier() {
  MPI_Barrier(MPI_COMM_WORLD);
}

CommType MPICommunicator::GetCommType() {
  return MPI;
}
}  // namespace net
}  // namespace cylon

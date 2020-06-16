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

#include "../communicator.h"
#include "mpi.h"
#include "mpi_communicator.h"
#include "mpi_channel.hpp"

namespace twisterx {
namespace net {
// configs
void MPIConfig::DummyConfig(int dummy) {
  this->AddConfig("Dummy", &dummy);
}
int MPIConfig::GetDummyConfig() {
  return *(int *) this->GetConfig("Dummy");
}

CommType MPIConfig::Type() {
  return CommType::MPI;
}

Channel *MPICommunicator::CreateChannel() {
  return new MPIChannel();
}

int MPICommunicator::GetRank() {
  return this->rank;
}
int MPICommunicator::GetWorldSize() {
  return this->world_size;
}
void MPICommunicator::Init(CommConfig *config) {
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    int provided;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
  } else {
    LOG(INFO) << "MPI is already initialized";
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &this->world_size);
}
void MPICommunicator::Finalize() {
  MPI_Finalize();
}
}
}
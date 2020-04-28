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
  MPI_Init(nullptr, nullptr);
  MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &this->world_size);
}
}
}
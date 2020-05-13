#ifndef TWISTERX_SRC_TWISTERX_COMM_MPICOMMUNICATOR_H_
#define TWISTERX_SRC_TWISTERX_COMM_MPICOMMUNICATOR_H_
#include "../comm_config.h"
#include "../communicator.h"
namespace twisterx {
namespace net {

class MPIConfig : public CommConfig {
  // no configs for MPI. This is an example
  void DummyConfig(int dummy);

  int GetDummyConfig();

  CommType Type();

};

class MPICommunicator : public Communicator {
  void Init(CommConfig *config);
  Channel *CreateChannel();
  int GetRank();
  int GetWorldSize();
  void Finalize();
};
}
}
#endif //TWISTERX_SRC_TWISTERX_COMM_MPICOMMUNICATOR_H_

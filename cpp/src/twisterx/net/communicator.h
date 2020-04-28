#ifndef TWISTERX_SRC_TWISTERX_COMM_COMMUNICATOR_H_
#define TWISTERX_SRC_TWISTERX_COMM_COMMUNICATOR_H_

#include "comm_config.h"
#include "channel.hpp"
namespace twisterx {
namespace net {

class Communicator {

 protected:
  int rank = -1;
  int world_size = -1;
 public:
  virtual void Init(CommConfig *config) = 0;
  virtual Channel *CreateChannel() = 0;
  virtual int GetRank() = 0;
  virtual int GetWorldSize() = 0;
};
}
}

#endif //TWISTERX_SRC_TWISTERX_COMM_COMMUNICATOR_H_

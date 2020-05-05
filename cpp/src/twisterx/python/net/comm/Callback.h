//
// Created by vibhatha on 4/27/20.
//

#ifndef TWISTERX_SRC_TWISTERX_PYTHON_NET_COMM_CALLBACK_H_
#define TWISTERX_SRC_TWISTERX_PYTHON_NET_COMM_CALLBACK_H_

#include "../../../net/ops/all_to_all.hpp"

namespace twisterx {
namespace net {
namespace comms {
class Callback : public twisterx::ReceiveCallback {
 public:
  bool onReceive(int source, void *buffer, int length);

  bool onReceiveHeader(int source, int finished, int *buffer, int length);

  bool onSendComplete(int target, void *buffer, int length);
};
}
}
}

#endif //TWISTERX_SRC_TWISTERX_PYTHON_NET_COMM_CALLBACK_H_

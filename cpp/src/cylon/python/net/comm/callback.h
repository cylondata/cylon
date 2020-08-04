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

#ifndef CYLON_SRC_CYLON_PYTHON_NET_COMM_CALLBACK_H_
#define CYLON_SRC_CYLON_PYTHON_NET_COMM_CALLBACK_H_

#include "net/ops/all_to_all.hpp"

namespace cylon {
namespace net {
namespace comms {
class Callback : public cylon::ReceiveCallback {
 public:
  bool onReceive(int source, void *buffer, int length);

  bool onReceiveHeader(int source, int finished, int *buffer, int length);

  bool onSendComplete(int target, void *buffer, int length);
};
}  // namespace comm
}  // namespace net
}  // namespace cylon

#endif //CYLON_SRC_CYLON_PYTHON_NET_COMM_CALLBACK_H_

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

#include "gloo_channel.hpp"

namespace cylon {
namespace net {

void GlooChannel::init(int edge,
                       const std::vector<int> &receives,
                       const std::vector<int> &sendIds,
                       ChannelReceiveCallback *rcv,
                       ChannelSendCallback *send,
                       Allocator *alloc) {

}
int GlooChannel::send(std::shared_ptr<TxRequest> request) {
  return 0;
}
int GlooChannel::sendFin(std::shared_ptr<TxRequest> request) {
  return 0;
}
void GlooChannel::progressSends() {

}
void GlooChannel::progressReceives() {

}
void GlooChannel::close() {

}
}
}
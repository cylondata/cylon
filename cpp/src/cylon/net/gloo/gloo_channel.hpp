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
#ifndef CYLON_CPP_SRC_CYLON_NET_GLOO_GLOO_CHANNEL_HPP_
#define CYLON_CPP_SRC_CYLON_NET_GLOO_GLOO_CHANNEL_HPP_

#include "cylon/net/channel.hpp"

namespace cylon {
namespace net {

class GlooChannel : public Channel {
 public:
  void init(int edge,
            const std::vector<int> &receives,
            const std::vector<int> &sendIds,
            ChannelReceiveCallback *rcv,
            ChannelSendCallback *send,
            Allocator *alloc) override;
  int send(std::shared_ptr<TxRequest> request) override;
  int sendFin(std::shared_ptr<TxRequest> request) override;
  void progressSends() override;
  void progressReceives() override;
  void close() override;
};

}
}

#endif //CYLON_CPP_SRC_CYLON_NET_GLOO_GLOO_CHANNEL_HPP_

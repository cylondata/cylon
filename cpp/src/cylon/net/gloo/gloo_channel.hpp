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

#include <unordered_map>
#include <queue>

#include <gloo/transport/unbound_buffer.h>
#include <gloo/context.h>

#include "cylon/net/channel.hpp"

namespace cylon {
namespace net {

enum SendStatus {
  SEND_INIT = 0,
  SEND_LENGTH_POSTED = 1,
  SEND_POSTED = 2,
  SEND_FINISH = 3,
  SEND_DONE = 4
};

enum ReceiveStatus {
  RECEIVE_INIT = 0,
  RECEIVE_LENGTH_POSTED = 1,
  RECEIVE_POSTED = 2,
  RECEIVED_FIN = 3
};

/**
 * Keep track about the length buffer to receive the length first
 */
struct PendingSend {
  //  we allow upto 8 ints for the header
  int header_buf[CYLON_CHANNEL_HEADER_SIZE]{};
  std::queue<std::shared_ptr<CylonRequest>> pending_data;
  SendStatus status = SEND_INIT;
  // the current send, if it is a actual send
  std::shared_ptr<CylonRequest> current_send{};

  std::unique_ptr<::gloo::transport::UnboundBuffer> request;
};

struct PendingReceive {
  // we allow upto 8 integer header
  int header_buf[CYLON_CHANNEL_HEADER_SIZE]{};
  int recv_id{};
  std::shared_ptr<Buffer> data{};
  int length{};
  ReceiveStatus status = RECEIVE_INIT;

  std::unique_ptr<gloo::transport::UnboundBuffer> request;
};

class GlooChannel : public Channel {
 public:
  explicit GlooChannel(gloo::Context *ctx_ptr);

  void init(int edge,
            const std::vector<int> &receives,
            const std::vector<int> &sendIds,
            ChannelReceiveCallback *rcv,
            ChannelSendCallback *send,
            Allocator *alloc) override;
  int send(std::shared_ptr<CylonRequest> request) override;
  int sendFin(std::shared_ptr<CylonRequest> request) override;
  void progressSends() override;
  void progressReceives() override;
  void close() override;

 private:
  gloo::Context *ctx_ptr;
  int rank;
  int edge_ = -1;
  // receive callback function
  ChannelReceiveCallback *rcv_fn = nullptr;
  // send complete callback function
  ChannelSendCallback *send_comp_fn = nullptr;
  // allocator
  Allocator *allocator = nullptr;

  // keep track of the length buffers for each receiver
  std::unordered_map<int, PendingSend> sends;
  // keep track of the posted receives
  std::unordered_map<int, PendingReceive> pending_receives;
  // we got finish requests
  std::unordered_map<int, std::shared_ptr<CylonRequest>> finish_requests;

  void sendHeader(std::pair<const int, PendingSend> &x) const;
  void sendFinishHeader(std::pair<const int, PendingSend> &x) const;
};

}
}

#endif //CYLON_CPP_SRC_CYLON_NET_GLOO_GLOO_CHANNEL_HPP_



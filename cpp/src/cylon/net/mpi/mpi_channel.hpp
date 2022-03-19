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

#ifndef CYLON_MPI_CHANNEL_H
#define CYLON_MPI_CHANNEL_H

#include <vector>
#include <unordered_map>
#include <queue>
#include <mpi.h>

#include <cylon/net/channel.hpp>
#include <cylon/net/buffer.hpp>

#define CYLON_CHANNEL_HEADER_SIZE 8
#define CYLON_MSG_FIN 1

namespace cylon {
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
  int headerBuf[CYLON_CHANNEL_HEADER_SIZE]{};
  std::queue<std::shared_ptr<CylonRequest>> pendingData;
  SendStatus status = SEND_INIT;
  MPI_Request request{};
  // the current send, if it is a actual send
  std::shared_ptr<CylonRequest> currentSend{};
};

struct PendingReceive {
  // we allow upto 8 integer header
  int headerBuf[CYLON_CHANNEL_HEADER_SIZE]{};
  int receiveId{};
  std::shared_ptr<Buffer> data{};
  int length{};
  ReceiveStatus status = RECEIVE_INIT;
  MPI_Request request{};
};

/**
 * This class implements a MPI channel, when there is a message to be sent,
 * this channel sends a small message with the size of the next message. This allows the other side
 * to post the network buffer to receive the message
 */
class MPIChannel : public Channel {
 public:
  explicit MPIChannel(MPI_Comm comm) : comm_(comm) {}

  /**
   * Initialize the channel
   *
   * @param receives receive from these ranks
   */
  void init(int edge, const std::vector<int> &receives, const std::vector<int> &sendIds,
            ChannelReceiveCallback *rcv, ChannelSendCallback *send, Allocator *alloc) override;

  /**
  * Send the message to the target.
  *
  * @param request the request
  * @return true if accepted
  */
  int send(std::shared_ptr<CylonRequest> request) override;

  /**
  * Send the message to the target.
  *
  * @param request the request
  * @return true if accepted
  */
  int sendFin(std::shared_ptr<CylonRequest> request) override;

  /**
   * This method, will send the messages, It will first send a message with length and then
   */
  void progressSends() override;

  /**
   * Progress the pending receivers
   */
  void progressReceives() override;

  void close() override;

  ~MPIChannel() override = default;

 private:
  int edge;
  // keep track of the length buffers for each receiver
  std::unordered_map<int, PendingSend *> sends;
  // keep track of the posted receives
  std::unordered_map<int, PendingReceive *> pendingReceives;
  // we got finish requests
  std::unordered_map<int, std::shared_ptr<CylonRequest>> finishRequests;
  // receive callback function
  ChannelReceiveCallback *rcv_fn;
  // send complete callback function
  ChannelSendCallback *send_comp_fn;
  // allocator
  Allocator *allocator;
  // mpi rank
  int rank;
  MPI_Comm comm_;

  /**
   * Send finish request
   * @param x the target, pendingSend pair
   */
  void sendFinishHeader(const std::pair<const int, PendingSend *> &x) const;

  /**
   * Send the length
   * @param x the target, pendingSend pair
   */
  void sendHeader(const std::pair<const int, PendingSend *> &x) const;
};
}

#endif
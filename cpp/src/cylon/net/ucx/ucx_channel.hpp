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

#ifndef CYLON_UCX_CHANNEL_H
#define CYLON_UCX_CHANNEL_H

#include <vector>
#include <unordered_map>
#include <queue>
#include <ucp/api/ucp.h>

#include <cylon/net/channel.hpp>
#include <cylon/net/ucx/ucx_operations.hpp>
#include <cylon/net/ucx/ucx_communicator.hpp>
#include <cylon/net/buffer.hpp>
#include <cylon/status.hpp>


#define CYLON_CHANNEL_HEADER_SIZE 8
#define CYLON_MSG_FIN 1

namespace cylon {
enum UCXSendStatus {
  SEND_INIT = 0,
  SEND_LENGTH_POSTED = 1,
  SEND_POSTED = 2,
  SEND_FINISH = 3,
  SEND_DONE = 4
};

enum UCXReceiveStatus {
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
  // segments of data to be sent
  std::queue<std::shared_ptr<TxRequest>> pendingData{};
  UCXSendStatus status = SEND_INIT;
  MPI_Request request{};
  // the current send, if it is a actual send
  std::shared_ptr<TxRequest> currentSend{};

  // UCX context - For tracking the progress of the message
  ucx::ucxContext *context;
};

struct PendingReceive {
  // we allow upto 8 integer header
  int headerBuf[CYLON_CHANNEL_HEADER_SIZE]{};
  int receiveId{};
  // Buffers are untyped: they simply denote a physical memory
  // area regardless of its intended meaning or interpretation.
  std::shared_ptr<Buffer> data{};
  int length{};
  UCXReceiveStatus status = RECEIVE_INIT;
  MPI_Request request{};
  // UCX context - For tracking the progress of the message
  ucx::ucxContext *context;
};

/**
 * This class implements a UCX channel, when there is a message to be sent,
 * this channel sends a small message with the size of the next message.
 * This allows the other side to post the network buffer to receive the message.
 */
class UCXChannel : public Channel {
 public:
  /**
   * Initialize the channel
   *
   * @param receives receive from these ranks
   */
  void init(int edge,
            const std::vector<int> &receives,
            const std::vector<int> &sendIds,
			ChannelReceiveCallback *rcv,
			ChannelSendCallback *send,
			Allocator *alloc) override;

  /**
  * Send the message to the target.
  *
  * @param request the request
  * @return true if accepted
  */
  int send(std::shared_ptr<TxRequest> request) override;

  /**
  * Send the message to the target.
  *
  * @param request the request
  * @return true if accepted
  */
  int sendFin(std::shared_ptr<TxRequest> request) override;

  /**
   * This method, will send the messages, It will first send a message with length and then
   */
  void progressSends() override;

  /**
   * Progress the pending receivers
   */
  void progressReceives() override;

  void close() override;

  /**
   * Link the necessary parameters associated with the communicator to the channel
   * @param [in] com - The UCX communicator that created the channel
   * @return
   */
  void linkCommunicator(net::UCXCommunicator* com);

 private:
  int edge;
  // keep track of the length buffers for each receiver
  std::unordered_map<int, PendingSend *> sends;
  // keep track of the posted receives
  std::unordered_map<int, PendingReceive *> pendingReceives;
  // we got finish requests
  std::unordered_map<int, std::shared_ptr<TxRequest>> finishRequests;
  // receive callback function
  ChannelReceiveCallback *rcv_fn;
  // send complete callback function
  ChannelSendCallback *send_comp_fn;
  // allocator
  Allocator *allocator;
  // mpi rank
  int rank;
  // mpi world size
  int worldSize;

  // # UCX specific attributes
  // The worker for receiving
  ucp_worker_h*  ucpRecvWorker;
  // The worker for sending
  ucp_worker_h*  ucpSendWorker;
  // Endpoint Map
  std::unordered_map<int, ucp_ep_h> endPointMap;
  // Tag mask used to match UCX send / receives
  ucp_tag_t tagMask = UINT64_MAX;

  /**
   * UCX Receive
   * Modeled after the IRECV function of MPI
   * @param [out] buffer - Pointer to the output buffer
   * @param [in] count - Size of the receiving data
   * @param [in] sender - MPI id of the sender
   * @param [out] ctx - ucx::ucxContext object, used for tracking the progress of the request
   * @return Cylon Status
   */
  Status UCX_Irecv(void *buffer,
                 size_t count,
                 int source,
                 ucx::ucxContext* ctx);

  /**
   * UCX Send
   * Modeled after the ISEND function of MPI
   * @param [out] buffer - Pointer to the buffer to send
   * @param [in] count - Size of the receiving data
   * @param [in] ep - Endpoint to send the data to
   * @param [out] request - UCX Context object
   *                        Used for tracking the progress of the request
   * @return Cylon Status
   */
  Status UCX_Isend(const void *buffer,
                 size_t  count,
                 ucp_ep_h ep,
                 ucx::ucxContext* request) const;

  /**
   * Send finish request
   * @param x the target, pendingSend pair
   */
  void sendFinishHeader(const std::pair<const int,
                        PendingSend *> &x) const;

  /**
   * Send the length
   * @param x the target, pendingSend pair
   */
  void sendHeader(const std::pair<const int,
                  PendingSend *> &x) const;
};
}

#endif
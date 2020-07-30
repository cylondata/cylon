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

#ifndef CYLON_CHANNEL_H
#define CYLON_CHANNEL_H

#include <vector>
#include <memory>
#include <cstring>
#include "TxRequest.hpp"

namespace cylon {

/**
 * When a send is complete, this callback is called by the channel, it is the responsibility
 * of the operations to register this callback
 */
class ChannelSendCallback {
 public:
  virtual void sendComplete(std::shared_ptr<TxRequest> request) = 0;

  virtual void sendFinishComplete(std::shared_ptr<TxRequest> request) = 0;
};

/**
 * When a receive is complete, this method is called
 */
class ChannelReceiveCallback {
 public:
  virtual void receivedData(int receiveId, void *buffer, int length) = 0;

  virtual void receivedHeader(int receiveId, int finished, int *header, int headerLength) = 0;
};

/**
 * This is an interface to send messages using a particular channel, a channel
 * can be based on MPI, it can be a TCP channel or a UCX channel etc
 */
class Channel {
 public:
  /**
   * Initialize the channel with the worker ids from which we are going to receive
   *
   * @param receives these are the workers we are going to receive from
   */
  virtual void init(int edge, const std::vector<int> &receives, const std::vector<int> &sendIds,
					ChannelReceiveCallback *rcv, ChannelSendCallback *send) = 0;
  /**
   * Send the request
   * @param request the request containing buffer, destination etc
   * @return if the request is accepted to be sent
   */
  virtual int send(std::shared_ptr<TxRequest> request) = 0;

  /**
   * Inform the finish to the target
   * @param request the request
   * @return -1 if not accepted, 1 if accepted
   */
  virtual int sendFin(std::shared_ptr<TxRequest> request) = 0;

  /**
   * This method needs to be called to progress the send
   */
  virtual void progressSends() = 0;

  /**
   * This method needs to be called to progress the receives
   */
  virtual void progressReceives() = 0;

  /**
   * Close the channel and clear any allocated memory by the channel
   */
  virtual void close() = 0;

  virtual ~Channel() = default;
};
}  // namespace cylon

#endif
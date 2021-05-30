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

#include "ucx_channel.hpp"

// TODO Sandeepa Remove the unnecessary
#include <glog/logging.h>
#include <vector>
#include <iostream>
#include <cstring>
#include <memory>
#include <utility>
#include <status.hpp>
#include <cassert>
#include <cmath>
#include <algorithm>

#include "ucx_operations.hpp"

// TODO Sandeepa remove - was put to test if connections happen
#include <fstream>
#include <string>

namespace cylon {

/**
 * Handle the completion of a receive.
 * Set completed to 1 in @param request to indicate completion.
 * @param [in,out] request - Request pinter
 * @param [in] status - Status of the request
 * @param [in] info - Info of the request. (length and the sender tag)
 * @return
 */
static void recvHandler(void *request,
                         ucs_status_t status,
                         ucp_tag_recv_info_t *info) {
  auto *context = (ucx::ucxContext *) request;
  context->completed = 1;
  // TODO Sandeepa message to handle recv completion
}

/**
 * Handle the completion of a send.
 * Set completed to 1 in @param ctx to indicate completion.
 * @param [in] request - Request pinter
 * @param [in] status - Status of the request
 * @param [in,out] ctx - Context (user data) of the request
 * @return
 */
static void sendHandler(void *request,
                         ucs_status_t status,
                         void *ctx) {
  auto *context = (ucx::ucxContext *) ctx;
  context->completed = 1;
  // TODO Sandeepa message to handle send completion
}

/**
 * Generate the tag used in messaging
 * Used to identify the sending and receiving pairs, not just the edge
 * Each of the 3 bytes used to represent an int is used to represent the stakeholders and sequence
 * Most significant byte represents the edge, the second most significant byte represents the receiver
 *  and the least significant byte represents the sender
 * @param [in] edge - Edge / Sequence
 * @param [in] recver - Receiver of the message
 * @param [in] sender - Sender of the message
 * @return int the compound tag
 */
static int getTag(int edge,
                   int recver,
                   int sender) {
  return (int)(edge * std::pow(2, 8 * 2) + recver * std::pow(2, 8 * 1) + sender);
}

/**
 * Check if all of the array is true. Used for checking if all sends / receives are completed when exchanging
 *  UCX worker addresses
 * @param [in] arr - Array containing the bool values
 * @return bool - if all elements in the array are true or not
 */
bool checkArr(std::vector<bool> &arr){
  return std::all_of(arr.begin(), arr.end(), [](bool elem){ return elem; });
}

/**
 * UCX Receive
 * Modeled after the IRECV function of MPI
 * @param [out] buffer - Pointer to the output buffer
 * @param [in] count - Size of the receiving data
 * @param [in] sender - MPI id of the sender
 * @return ucx::ucxContext - Used for tracking the progress of the request
 */
ucx::ucxContext *UCXChannel::UCX_Irecv(void *buffer,
                                       size_t count,
                                       int sender) {
  // UCX context / request
  ucx::ucxContext *request;

  // UCP non-blocking tag receive
  // Inp - UCP worker, buffer, length, datatype, tag, tag mask, receive handler
  request = (ucx::ucxContext *) ucp_tag_recv_nb(*ucpRecvWorker,
                                            buffer,
                                            count,
                                            ucp_dt_make_contig(1),
                                            getTag(edge, rank, sender),
                                            tagMask,
                                            recvHandler);

  // Check if there is an error in the request
  if (UCS_PTR_IS_ERR(request)) {
    LOG(FATAL) << "Unable to receive UCX message " <<
               ucs_status_string(UCS_PTR_STATUS(request));
    return nullptr;
  } else {
    assert(UCS_PTR_IS_PTR(request));
    return request;
  }
}

/**
 * UCX Send
 * Modeled after the ISEND function of MPI
 * @param [out] buffer - Pointer to the buffer to send
 * @param [in] count - Size of the receiving data
 * @param [in] ep - Endpoint to send the data to
 * @param [in] target - MPI id of the receiver / target
 * @param [out] ctx - Used for tracking the progress of the request
 */
void UCXChannel::UCX_Isend(const void *buffer,
                           size_t count,
                           ucp_ep_h ep,
                           int target,
                           ucx::ucxContext* ctx) const {
  // To hold the status of operations
  ucs_status_ptr_t status;

  // Send parameters (Mask, callback, context)
  ucp_request_param_t sendParam;
  sendParam.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
      UCP_OP_ATTR_FIELD_USER_DATA |
      UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
  sendParam.cb.send = sendHandler;
  sendParam.user_data = ctx;

  // Init completed
  ctx->completed = 0;
  // UCP non-blocking tag send
  // Inp - Endpoint, buffer, length, tag, send parameters
  status = ucp_tag_send_nbx(ep,
                            buffer,
                            count,
                            getTag(edge, target, rank),
                            &sendParam);
  // Check if there is an error in the request
  if (UCS_PTR_IS_ERR(status)) {
    LOG(FATAL) << "Error in sending message via UCX";
  }
  // Handle the situation where the ucp_tag_send_nbx function returns immediately
  // without calling the send handler
  if (!UCS_PTR_IS_PTR(status) && status == nullptr) {
    ctx->completed = 1;
  }
}

/**
 * Initialize the UCX network by sending / receiving the UCX worker addresses via MPI
 * @param [in] receives - MPI IDs of the nodes to receive from
 * @param [in] sendIds - MPI IDs of the nodes to send to
 * @return
 */
void UCXChannel::linkCommunicator(net::UCXCommunicator *com) {
  this->ucxCom = com;

  this->worldSize = com->GetWorldSize();
  this->rank = com->GetRank();
  this->ucpRecvWorker = &(com->ucpRecvWorker);
  this->ucpSendWorker = &(com->ucpSendWorker);

  for (auto x : com->endPointMap) {
    ucp_ep_h ep = x.second;
    this->endPointMap[x.first] = ep;
  }
}

// *********************** Implementations of UCX Channel ***********************
/**
 * Initialize UCX Channel for Cylon
 * @param [in]  ed
 * @param [in]  receives
 * @param [in]  sendIds
 * @param [in]  rcv
 * @param [in]  send_fn
 * @param [in]  alloc
 */
void UCXChannel::init(int ed,
                      const std::vector<int> &receives,
                      const std::vector<int> &sendIds,
                      ChannelReceiveCallback *rcv,
                      ChannelSendCallback *send_fn,
                      Allocator *alloc) {
  // Storing the parameters given by the Cylon Channel class
  edge = ed;
  rcv_fn = rcv;
  send_comp_fn = send_fn;
  allocator = alloc;

  // TODO Sandeepa code for sending the worker address via sockets moved to extra_code.txt
  // socketInit(receives, sendIds);
  // MPIInit(receives, sendIds);

  // Get the number of receives and sends to be used in iterations
  int numReci = (int) receives.size();
  int numSends = (int) sendIds.size();
  // Int variable used when iterating
  int sIndx;

  // Iterate and set the receives
  for (sIndx = 0; sIndx < numReci; sIndx++) {
    // Rank of the node receiving from
    int recvRank = receives.at(sIndx);
    // Init a new pending receive for the request
    auto *buf = new PendingReceive();
    // TODO Sandeepa remove ipInt allocation since it's not used anywhere
    buf->receiveId = recvRank;
    // Add to pendingReceive object to pendingReceives map
    pendingReceives.insert(std::pair<int, PendingReceive *>(recvRank, buf));
    // Receive for the initial header buffer
    buf->context = UCX_Irecv(buf->headerBuf,
                             CYLON_CHANNEL_HEADER_SIZE * sizeof(int),
                             recvRank);
    // Init status of the receive
    buf->status = RECEIVE_LENGTH_POSTED;
  }


  // Iterate and set the sends
  for (sIndx = 0; sIndx < numSends; sIndx++) {
    // Rank of the node sending to
    int sendRank = sendIds.at(sIndx);
    // Init a new pending send for the request
    sends[sendRank] = new PendingSend();
  }

  // TODO Sandeepa remove
  const char *path="file.txt";
  std::ofstream file(path); //open in constructor
  std::string data("data to write to file");
  file << data;
}

/**
  * Send the request
  * @param request the request containing buffer, destination etc
  * @return if the request is accepted to be sent
  */
int UCXChannel::send(std::shared_ptr<TxRequest> request) {
  // Loads the pending send from sends
  PendingSend *ps = sends[request->target];
  if (ps->pendingData.size() > 1000) {
    return -1;
  }
  // pendingData is a queue that has TXRequests
  ps->pendingData.push(request);
  return 1;
}

/**
  * Inform the finish to the target
  * @param request the request
  * @return -1 if not accepted, 1 if accepted
  */
int UCXChannel::sendFin(std::shared_ptr<TxRequest> request) {
  // Checks if the finished request is alreay in finished req
  // If so, give error
  if (finishRequests.find(request->target) != finishRequests.end()) {
    return -1;
  }

  // Add finished req to map
  finishRequests.insert(std::pair<int, std::shared_ptr<TxRequest>>(request->target, request));
  return 1;
}

void UCXChannel::progressReceives() {
  // Progress the ucp receive worker
  ucp_worker_progress(*ucpRecvWorker);

  // Iterate through the pending receives
  for (auto x : pendingReceives) {
    // Check if the buffer is posted
    if (x.second->status == RECEIVE_LENGTH_POSTED) {
      // If completed request is completed
      if (x.second->context->completed == 1) {
        // Release the existing context
//        if (x.second->context != nullptr){
//          ucp_request_release(x.second->context);
//        }

        // TODO Sandeepa is the count check necessary
        //  Can be done by ucp_tag_probe_nb?
        // The number of received elements
//        int count = 8;

        // Get data from the header
        // read the length from the header
        int length = x.second->headerBuf[0];
        int finFlag = x.second->headerBuf[1];

        // Check weather we are at the end
        if (finFlag != CYLON_MSG_FIN) {
          // If not at the end

          // Malloc a buffer
          Status stat = allocator->Allocate(length, &x.second->data);
          if (!stat.is_ok()) {
            LOG(FATAL) << "Failed to allocate buffer with length " << length;
          }

          // Set the length
          x.second->length = length;
          // UCX receive
          x.second->context = UCX_Irecv(x.second->data->GetByteBuffer(), length, x.first);
          // Set the flag to true so we can identify later which buffers are posted
          x.second->status = RECEIVE_POSTED;

          // copy the count - 2 to the buffer
          int *header = nullptr;
          header = new int[6];
          memcpy(header, &(x.second->headerBuf[2]), 6 * sizeof(int));

          // Notify the receiver that the destination received the header
          rcv_fn->receivedHeader(x.first, finFlag, header, 6);
        } else {
          // We are not expecting to receive any more
          x.second->status = RECEIVED_FIN;
          // Notify the receiver
          rcv_fn->receivedHeader(x.first, finFlag, nullptr, 0);
        }
      }
    } else if (x.second->status == RECEIVE_POSTED) {
      // if request completed
      if (x.second->context->completed == 1) {
        // Release the existing context
//        if (x.second->context != nullptr){
//          ucp_request_release(x.second->context);
//        }

        // Fill header buffer
        std::fill_n(x.second->headerBuf, CYLON_CHANNEL_HEADER_SIZE, 0);

        // UCX receive
        x.second->context = UCX_Irecv(x.second->headerBuf,
                                      CYLON_CHANNEL_HEADER_SIZE * sizeof(int),
                                      x.first);
        // Set state
        x.second->status = RECEIVE_LENGTH_POSTED;
        // Call the back end
        rcv_fn->receivedData(x.first, x.second->data, x.second->length);
      }
      // TODO Sandeepa would a early return have any advantages? (There would be a string of jumps or a single jump?)
    } else if (x.second->status != RECEIVED_FIN) {
      LOG(FATAL) << "At an un-expected state " << x.second->status;
    }
  }
}

void UCXChannel::progressSends() {
  // Progress the ucp send worker
  ucp_worker_progress(*ucpSendWorker);

  // Iterate through the sends
  for (auto x : sends) {
    // If currently in the length posted stage of the send
    if (x.second->status == SEND_LENGTH_POSTED) {
      // If completed
      if (x.second->context->completed == 1) {
        // Destroy context object
        //  NOTE can't use ucp_request_release here cuz we actually init our own UCX context here
        delete x.second->context;

        // Post the actual send
        std::shared_ptr<TxRequest> r = x.second->pendingData.front();
        // Send the message
        x.second->context =  new ucx::ucxContext();
        UCX_Isend(r->buffer,
                  r->length,
                  endPointMap[x.first],
                  x.first,
                  x.second->context);

        // Update status
        x.second->status = SEND_POSTED;

        // We set to the current send and pop it
        x.second->pendingData.pop();
        // The update the current send in the queue of sends
        x.second->currentSend = r;
      }
    } else if (x.second->status == SEND_INIT) {
      // Send header if no pending data
      if (!x.second->pendingData.empty()) {
        sendHeader(x);
      } else if (finishRequests.find(x.first) != finishRequests.end()) {
        // If there are finish requests lets send them
        sendFinishHeader(x);
      }
    } else if (x.second->status == SEND_POSTED) {
      // If completed
      if (x.second->context->completed == 1) {
        // If there are more data to post, post the length buffer now
        if (!x.second->pendingData.empty()) {
          // If the pending data is not empty
          sendHeader(x);
          // We need to notify about the send completion
          send_comp_fn->sendComplete(x.second->currentSend);
          x.second->currentSend = {};
        } else {
          // If pending data is empty
          // Notify about send completion
          send_comp_fn->sendComplete(x.second->currentSend);
          x.second->currentSend = {};

          // Check if request is in finish
          if (finishRequests.find(x.first) != finishRequests.end()) {
            sendFinishHeader(x);
          } else {
            // If req is not in finish then re-init
            x.second->status = SEND_INIT;
          }
        }
      }
    } else if (x.second->status == SEND_FINISH) {
      if (x.second->context->completed == 1) {
        // We are going to send complete
        std::shared_ptr<TxRequest> finReq = finishRequests[x.first];
        send_comp_fn->sendFinishComplete(finReq);
        x.second->status = SEND_DONE;
      }
    } else if (x.second->status != SEND_DONE) {
      // If an unknown state
      // Throw an exception and log
      LOG(FATAL) << "At an un-expected state " << x.second->status;
    }
  }
}

/**
 * Send the length
 * @param x the target, pendingSend pair
 */
void UCXChannel::sendHeader(const std::pair<const int, PendingSend *> &x) const {
  // Get the request
  std::shared_ptr<TxRequest> r = x.second->pendingData.front();
  // Put the length to the buffer
  // TODO Sandeepa Is it possible to reduce the headers?
  x.second->headerBuf[0] = r->length;
  x.second->headerBuf[1] = 0;

  // Copy data from TxRequest header to the PendingSend header
  if (r->headerLength > 0) {
    memcpy(&(x.second->headerBuf[2]),
           &(r->header[0]),
           r->headerLength * sizeof(int));
  }
  delete x.second->context;
  // UCX send of the header
  x.second->context =  new ucx::ucxContext();
  UCX_Isend(x.second->headerBuf,
            (2 + r->headerLength) * sizeof(int),
            endPointMap.at(x.first),
            x.first,
            x.second->context);
  // Update status
  x.second->status = SEND_LENGTH_POSTED;
}

/**
 * Send the length
 * @param x the target, pendingSend pair
 */
void UCXChannel::sendFinishHeader(const std::pair<const int, PendingSend *> &x) const {
  // for the last header we always send only the first 2 integers
  x.second->headerBuf[0] = 0;
  x.second->headerBuf[1] = CYLON_MSG_FIN;
  // TODO Sandeepa count was 2, changed to 8 since 8 is hardcoded atm
  delete x.second->context;
  x.second->context =  new ucx::ucxContext();
  UCX_Isend(x.second->headerBuf,
            8 * sizeof(int),
            endPointMap.at(x.first),
            x.first,
            x.second->context);
  x.second->status = SEND_FINISH;
}

/**
 * Close the channel and clear any allocated memory by the channel
 */
void UCXChannel::close() {
  // TODO Sandeepa add all cleanups
  // Clear pending receives
  for (auto &pendingReceive : pendingReceives) {
    delete (pendingReceive.second);
  }
  pendingReceives.clear();

  // Clear the sends
  for (auto &s : sends) {
    delete (s.second);
  }
  sends.clear();
}
}  // namespace cylon

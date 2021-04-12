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
struct ucx::ucxContext *UCXChannel::UCX_Irecv(void *buffer,
                                       size_t count,
                                       int sender) {
  // UCX context / request
  struct ucx::ucxContext *request;

  // UCP non-blocking tag receive
  // Inp - UCP worker, buffer, length, datatype, tag, tag mask, receive handler
  request = (ucx::ucxContext *) ucp_tag_recv_nb(ucpRecvWorker,
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
 * @return ucx::ucxContext - Used for tracking the progress of the request
 */
struct ucx::ucxContext *UCXChannel::UCX_Isend(const void *buffer,
                                       size_t count,
                                       ucp_ep_h ep,
                                       int target) const {
  // A new context to handle the current send
  auto *ctx = new ucx::ucxContext();
  // To hold the status of operations
  ucs_status_ptr_t status;

  // Send parameters (Mask, callback, context)
  ucp_request_param_t sendParam;
  sendParam.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
      UCP_OP_ATTR_FIELD_USER_DATA;
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
    return nullptr;
  }
  // Handle the situation where the ucp_tag_send_nbx function returns immediately
  // without calling the send handler
  if (!UCS_PTR_IS_PTR(status)) {
    ctx->completed = 1;
  }
  return ctx;
}

/**
 * Initialize the UCX network by sending / receiving the UCX worker addresses via MPI
 * @param [in] receives - MPI IDs of the nodes to receive from
 * @param [in] sendIds - MPI IDs of the nodes to send to
 * @return
 */
void UCXChannel::MPIInit(const std::vector<int> &receives,
                            const std::vector<int> &sendIds) {
  // Hold return value of functions
  int ret;
  // Get the rank for checking send to self, and initializations
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // UCP Context - Holds a UCP communication instance's global information.
  ucp_context_h ucpContext;
  // TODO Sandeepa have a common distributed context?
  // Init context
  ret = cylon::ucx::initContext(&ucpContext, nullptr);
  if (ret != 0) {
    LOG(FATAL) << "Error occurred when creating UCX context";
  }

  // UCP Worker - The worker represents an instance of a local communication resource
  // and the progress engine associated with it.
  // Init recv worker
  ucpRecvWorkerAddr = cylon::ucx::initWorker(ucpContext,
                                             &ucpRecvWorker);
  // Init send worker
  cylon::ucx::initWorker(ucpContext,
                         &ucpSendWorker);

  int numReci = (int) receives.size();
  int numSends = (int) sendIds.size();
  int sIndx;

  // Iterate and set the receives
  for (sIndx = 0; sIndx < numReci; sIndx++) {
    int ipInt = receives.at(sIndx);
    auto *buf = new PendingReceive();

    if (ipInt != rank) {
      MPI_Isend(ucpRecvWorkerAddr->addr,
                (int)ucpRecvWorkerAddr->addrSize,
                MPI_BYTE,
                ipInt,
                edge,
                MPI_COMM_WORLD,
                &(buf->request));
    }
    buf->receiveId = ipInt;
    pendingReceives.insert(std::pair<int, PendingReceive *>(ipInt, buf));
    buf->context = UCX_Irecv(buf->headerBuf,
                             CYLON_CHANNEL_HEADER_SIZE * sizeof(int),
                             ipInt);
    buf->status = RECEIVE_LENGTH_POSTED;
  }


  // Iterate and set the sends
  for (sIndx = 0; sIndx < numSends; sIndx++) {
    int ipInt = sendIds.at(sIndx);
    auto w2 = new ucx::ucxWorker();

    sends[ipInt] = new PendingSend();
    // TODO Sandeepa remove
//    std::cout << "Send size: " << sends[ipInt]->pendingData.size() << std::endl;

    sends[ipInt]->wa = w2;
    w2->addrSize = ucpRecvWorkerAddr->addrSize;

    if (ipInt != rank) {
      w2->addr = (ucp_address_t*)malloc(ucpRecvWorkerAddr->addrSize);
      // TODO Sandeepa length is set based on the local address set
      MPI_Irecv(w2->addr,
                (int)w2->addrSize,
                MPI_BYTE,
                ipInt,
                edge,
                MPI_COMM_WORLD,
                &(sends[ipInt]->request));
    }
  }

  std::vector<bool> finRecv (numReci, false);
  std::vector<bool> finSend (numSends, false);

  int flag;
  int ipInt;
  MPI_Status status;

  while (true){
    for (sIndx = 0; sIndx < numReci; sIndx++) {
      ipInt = receives.at(sIndx);
      flag = 0;
      status = {};

      if (ipInt != rank) {
        MPI_Test(&(pendingReceives[ipInt]->request), &flag, &status);
      } else {
        flag = 1;
      }
      if (!finRecv[sIndx] && flag) {
        finRecv[sIndx]=true;
      }
    }

    for (sIndx = 0; sIndx < numSends; sIndx++) {
      ipInt = receives.at(sIndx);
      flag = 0;
      status = {};
      if (ipInt != rank) {
        MPI_Test(&(sends[ipInt]->request), &flag, &status);
      } else {
        flag = 1;
      }

      if (!finSend[sIndx] && flag) {
        ucs_status_t ucxStatus;
        ucp_ep_params_t epParams;

        if (ipInt != rank) {
          if (sends[ipInt]->wa->addr == nullptr) {
            std::cerr << "Error when receiving send worker address" << std::endl;
            return;
          }
        } else {
          sends[ipInt]->wa->addr = ucpRecvWorkerAddr->addr;
          sends[ipInt]->wa->addrSize = ucpRecvWorkerAddr->addrSize;
        }

        /* Send client UCX address to server */
        epParams.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
            UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
        epParams.address = sends[ipInt]->wa->addr;
        epParams.err_mode = UCP_ERR_HANDLING_MODE_NONE;

        ucxStatus = ucp_ep_create(ucpSendWorker,
                                  &epParams,
                                  &sends[ipInt]->wa->ep);

        if (ucxStatus != UCS_OK) {
          std::cerr
              << "Error when creating the endpoint."
              << std::endl;
        }
        finSend[sIndx]=true;
      }
    }

    if(checkArr(finRecv) && checkArr(finSend)){
      break;
    }
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
  MPIInit(receives, sendIds);
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
  ucp_worker_progress(ucpRecvWorker);

  // Iterate through the pending recvs
  for (auto x : pendingReceives) {
    // check if the buffer is posted
    if (x.second->status == RECEIVE_LENGTH_POSTED) {
      // If completed request is completed
      if (x.second->context->completed == 1) {
//        x.second->context = new ucx::ucxContext();
        // TODO destroy existing context object
//        ucp_request_release(x.second->context);
        if (x.second->context != nullptr){
          ucp_request_release(x.second->context);
        }

        // TODO Sandeepa remove mpi request release
//        x.second->request = {};

        // TODO Sandeepa find out what count is used for
        // The number of received elements
        int count = 8;

        // TODO Sandeepa get count how?
//        MPI_Get_count(&status, MPI_INT, &count);

        // Get data from the header
        // read the length from the header
        int length = x.second->headerBuf[0];
        int finFlag = x.second->headerBuf[1];

        // check weather we are at the end
        // TODO Sandeepa ^ end as in currently progressing the data rather than the header which was sent earlier?
        if (finFlag != CYLON_MSG_FIN) {
          // If not at the end

          // Check if the header size is exceeded
          // TODO Sandeepa check count somehow?
//          if (count > 8) {
//            LOG(FATAL) << "Un-expected number of bytes expected: 8 or less "
//                       << " received: " << count;
//          }

          // malloc a buffer
          Status stat = allocator->Allocate(length, &x.second->data);
          if (!stat.is_ok()) {
            LOG(FATAL) << "Failed to allocate buffer with length " << length;
          }

          // TODO Sandeepa the length wasn't specified earlier?
          x.second->length = length;

          // TODO Sandeepa why create a new receive request here? Does CYLON_MSG_FIN mean that the buffer was posted?
          x.second->context = UCX_Irecv(x.second->data->GetByteBuffer(), length, x.first);

          // set the flag to true so we can identify later which buffers are posted
          x.second->status = RECEIVE_POSTED;

          // TODO Sandeepa why was the buffer reduced by 2? Does the header when putting in receivedHeader
          // copy the count - 2 to the buffer
          int *header = nullptr;
          if (count > 2) {
            header = new int[count - 2];
            memcpy(header, &(x.second->headerBuf[2]), (count - 2) * sizeof(int));
          }

          // Notify the receiver that the destination received the header
          rcv_fn->receivedHeader(x.first, finFlag, header, count - 2);
        } else {
          // TODO Sandeepa is header count only 2 if data or the end of data?
//          if (count != 2) {
//            LOG(FATAL) << "Un-expected number of bytes expected: 2 " << " received: " << count;
//          }
          // we are not expecting to receive any more
          x.second->status = RECEIVED_FIN;
          // notify the receiver
          rcv_fn->receivedHeader(x.first, finFlag, nullptr, 0);
        }
      }
    } else if (x.second->status == RECEIVE_POSTED) {
      // if request completed
      if (x.second->context->completed == 1) {
        // The number of received elements
//        int count = 0;
//        MPI_Get_count(&status, MPI_BYTE, &count);

        // Check if the count is the number specified
        // TODO Sandeepa why is this check not used earlier? Is the length not set there?
        // TODO is this why length is set later on?
//        if (count != x.second->length) {
//          LOG(FATAL) << "Un-expected number of bytes expected:" << x.second->length
//                     << " received: " << count;
//        }

        // Reset request
//        x.second->request = {};
        // TODO Sandeepa check if this is valid at this point
        // TODO Sandeepa re-init request?
        x.second->context = new ucx::ucxContext();
        // TODO destroy existing context object
//        ucp_request_release(x.second->context);
        // clear the headerBuffer array
        std::fill_n(x.second->headerBuf, CYLON_CHANNEL_HEADER_SIZE, 0);

        x.second->context = UCX_Irecv(x.second->headerBuf, CYLON_CHANNEL_HEADER_SIZE * sizeof(int), x.first);
        x.second->status = RECEIVE_LENGTH_POSTED;
        // call the back end
        rcv_fn->receivedData(x.first, x.second->data, x.second->length);
      }
      // TODO Sandeepa would a early return have any advantages? (There would be a string of jumps or a single jump?)
    } else if (x.second->status != RECEIVED_FIN) {
      LOG(FATAL) << "At an un-expected state " << x.second->status;
    }
  }
}

void UCXChannel::progressSends() {
  // TODO Sandeepa uh?
  // lets send values
  // Iterate through the sends
  ucp_worker_progress(ucpSendWorker);
  for (auto x : sends) {

    // If currently in the length posted stage of the send
    if (x.second->status == SEND_LENGTH_POSTED) {
      // Tests for the completion of a request
      // true if operation completed and its status

      // if completed
      if (x.second->context->completed == 1) {

        // TODO Sandeepa re-init request?
        x.second->context = new ucx::ucxContext();
        // TODO destroy existing context object
//        ucp_request_release(x.second->context);


        // Post the actual send
        // TODO Sandeepa pendingData is a queue of TxReqs (Check a place that it is created)
        std::shared_ptr<TxRequest> r = x.second->pendingData.front();
        // Send the message
        x.second->context = UCX_Isend(r->buffer, r->length, x.second->wa->ep, x.first);

        // Update status
        x.second->status = SEND_POSTED;
        x.second->pendingData.pop();
        // we set to the current send and pop it
        // The update the current send in the queue of sends
        x.second->currentSend = r;
      }
    } else if (x.second->status == SEND_INIT) {
      // TODO Sandeepa
      //  When the status is SEND_INIT

      x.second->request = {};

      // TODO Sandeepa give back to if
      bool stat = x.second->pendingData.empty();
      // Send header if no pending data
      if (!stat) {
        sendHeader(x);
      } else if (finishRequests.find(x.first) != finishRequests.end()) {
        // TODO Sandeepa send the finished requests? When the status is SEND_INIT?
        // If there are finish requests lets send them
        sendFinishHeader(x);
      }
    } else if (x.second->status == SEND_POSTED) {
      // TODO Sandeepa this is during the data transfers

      // Tests for the completion of a request
      // true if operation completed and its status

      // If completed
      if (x.second->context->completed == 1) {
        // TODO Sandeepa re-init request?
        x.second->request = {};
        // if there are more data to post, post the length buffer now

        // Send header if no pending data
        if (!x.second->pendingData.empty()) {
          // If the pending data is not empty??
          sendHeader(x);
          // we need to notify about the send completion
          // TODO Sandeepa Callback for completing
          send_comp_fn->sendComplete(x.second->currentSend);
          x.second->currentSend = {};
        } else {
          // If pending data is empty

          // Notify about send completion
          send_comp_fn->sendComplete(x.second->currentSend);
          x.second->currentSend = {};

          // Check if request is in finish
          if (finishRequests.find(x.first) != finishRequests.end()) {
            // TODO Sandeepa WTH is the difference between sendFinishHeader and sendComplete
            sendFinishHeader(x);
          } else {
            // TODO Sandeepa why re-init the send?
            // If req is not in finish then re-init?
            x.second->status = SEND_INIT;
          }
        }
      }
    } else if (x.second->status == SEND_FINISH) {
      // TODO Sandeepa What is this state?

      if (x.second->context->completed == 1) {
        // we are going to send complete
        std::shared_ptr<TxRequest> finReq = finishRequests[x.first];
        // TODO Sandeepa Another finish? orz
        send_comp_fn->sendFinishComplete(finReq);
        x.second->status = SEND_DONE;
      }
    } else if (x.second->status != SEND_DONE) {
      // If an unknown state

      // TODO Sandeepa Exception not thrown?
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
  // TODO Sandeepa Get data from something and update the orignial source with the copy??
  // TODO Sandeepa Is it possible to reduce the headers?
  x.second->headerBuf[0] = r->length;
  x.second->headerBuf[1] = 0;

  // Copy the memory of the header
  // TODO Sandeepa copy data from TxRequest header to the PendingSend header
  if (r->headerLength > 0) {
    memcpy(&(x.second->headerBuf[2]), &(r->header[0]), r->headerLength * sizeof(int));
  }
  x.second->context = UCX_Isend(x.second->headerBuf,
                                (2 + r->headerLength) * sizeof(int),
                                x.second->wa->ep,
                                x.first);
  x.second->status = SEND_LENGTH_POSTED;
}

/**
 * Send the length
 * @param x the target, pendingSend pair
 * TODO Sandeepa the function called when finishing the send?
 */
void UCXChannel::sendFinishHeader(const std::pair<const int, PendingSend *> &x) const {
  // for the last header we always send only the first 2 integers
  x.second->headerBuf[0] = 0;
  x.second->headerBuf[1] = CYLON_MSG_FIN;
  // TODO Sandeepa count was 2, changed to 8 since 8 is hardcoded atm
  x.second->context = UCX_Isend(x.second->headerBuf,
                                8 * sizeof(int),
                                x.second->wa->ep,
                                x.first);
  x.second->status = SEND_FINISH;
}

/**
 * Close the channel and clear any allocated memory by the channel
 */
void UCXChannel::close() {
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

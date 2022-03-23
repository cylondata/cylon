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

#include <glog/logging.h>
#include <cassert>

#include "gloo_channel.hpp"

namespace cylon {
namespace net {

//static constexpr std::chrono::milliseconds kTimeout(10);

void GlooChannel::init(int edge,
                       const std::vector<int> &receives,
                       const std::vector<int> &sendIds,
                       ChannelReceiveCallback *rcv,
                       ChannelSendCallback *send,
                       Allocator *alloc) {
  edge_ = edge;
  rcv_fn = rcv;
  send_comp_fn = send;
  allocator = alloc;

  pending_receives.reserve(receives.size());
  sends.reserve(sendIds.size());

  // we need to post the length buffers
  for (int source: receives) {
    auto iter = pending_receives.emplace(source, PendingReceive{});
    auto &p_recv = iter.first->second;
    p_recv.recv_id = source;
    p_recv.request = ctx_ptr->createUnboundBuffer((void *) &p_recv.header_buf[0],
                                                  CYLON_CHANNEL_HEADER_SIZE * sizeof(int));
//
//    auto *buf = new PendingReceive();
//    buf->receiveId = source;
//    pending_receives.insert(std::pair<int, PendingReceive *>(source, buf));
//    MPI_Irecv(buf->headerBuf, CYLON_CHANNEL_HEADER_SIZE, MPI_INT,
//              source, edge, comm_, &buf->request);
    p_recv.request->recv(source, edge, 0, CYLON_CHANNEL_HEADER_SIZE * sizeof(int));
    // set the flag to true so we can identify later which buffers are posted
    p_recv.status = RECEIVE_LENGTH_POSTED;
  }

  for (int target: sendIds) {
    sends.emplace(target, PendingSend{});
  }
//  // get the rank
//  MPI_Comm_rank(comm_, &rank);

}

int GlooChannel::send(std::shared_ptr<CylonRequest> request) {
  auto &pending_data = sends[request->target].pending_data;
  if (pending_data.size() > MAX_PENDING) {
    return -1;
  }
  pending_data.emplace(std::move(request));
  return 0;
}

int GlooChannel::sendFin(std::shared_ptr<CylonRequest> request) {
  if (finish_requests.find(request->target) != finish_requests.end()) {
    // sendFin has already received for this target
    return -1;
  }

  finish_requests.emplace(request->target, std::move(request));
  return 1;
}

void GlooChannel::sendHeader(std::pair<const int, PendingSend> &x) const {
  int dest = x.first;
  auto &pend_send = x.second;
  const auto &r = *pend_send.pending_data.front();
  // put the length to the buffer
  pend_send.header_buf[0] = r.length;
  pend_send.header_buf[1] = 0;

  // copy the memory of the header
  if (r.headerLength > 0) {
    memcpy(&pend_send.header_buf[2], r.header, r.headerLength * sizeof(int));
  }

  // we have to add 2 to the header length
//  MPI_Isend(&(x.second->headerBuf[0]), 2 + r->headerLength, MPI_INT,
//            x.first, edge, comm_, &(x.second->request));
  pend_send.request =
      ctx_ptr->createUnboundBuffer(&pend_send.header_buf[0], (2 + r.headerLength) * sizeof(int));
  pend_send.request->send(dest, edge_, 0, (2 + r.headerLength) * sizeof(int));
  pend_send.status = SEND_LENGTH_POSTED;
}

void GlooChannel::sendFinishHeader(std::pair<const int, PendingSend> &x) const {
  int dest = x.first;
  auto &pend_send = x.second;
  // for the last header we always send only the first 2 integers
  pend_send.header_buf[0] = 0;
  pend_send.header_buf[1] = CYLON_MSG_FIN;
//  MPI_Isend(&(x.second->headerBuf[0]), 2, MPI_INT,
//            x.first, edge, comm_, &(x.second->request));
  pend_send.request = ctx_ptr->createUnboundBuffer(pend_send.header_buf, 2 * sizeof(int));
  pend_send.request->send(dest, edge_, 0, 2 * sizeof(int));
  pend_send.status = SEND_FINISH;
}

void GlooChannel::progressSends() {
  for (auto &x: sends) {
//    int flag = 0;
//    MPI_Status status;
    int dest = x.first;
    auto &pend_send = x.second;
    bool flag = false;
    // if we are in the length posted
    if (pend_send.status == SEND_LENGTH_POSTED) {
//      MPI_Test(&x.second->request, &flag, &status);
      flag = pend_send.request->waitSend();
      if (flag) {
//        x.second->request = {};
        // now post the actual send
//        std::shared_ptr<CylonRequest> r = pend_send.pending_data.front();
        pend_send.current_send = pend_send.pending_data.front();

        const auto &r = *pend_send.current_send;
        pend_send.request = ctx_ptr->createUnboundBuffer(const_cast<void *>(r.buffer), r.length);
        pend_send.request->send(r.target, edge_, 0, r.length);
//        MPI_Isend(r->buffer, r->length, MPI_BYTE,
//                  r->target, edge, comm_, &(x.second->request));
        pend_send.status = SEND_POSTED;
        pend_send.pending_data.pop();
        // we set to the current send and pop it
//        x.second->currentSend = r;
      }
    } else if (pend_send.status == SEND_INIT) {
//      x.second->request = {};
      // now post the actual send
//      if (!x.second->pendingData.empty()) {
      if (!pend_send.pending_data.empty()) {
        sendHeader(x);
      } else if (finish_requests.find(dest) != finish_requests.end()) {
        // if there are finish requests lets send them
        sendFinishHeader(x);
      }
    } else if (pend_send.status == SEND_POSTED) {
//      MPI_Test(&(x.second->request), &flag, &status);
      flag = pend_send.request->waitSend();
      if (flag) {
        pend_send.request = {};
        // if there are more data to post, post the length buffer now
        if (!pend_send.pending_data.empty()) {
          sendHeader(x);
          // we need to notify about the send completion
          send_comp_fn->sendComplete(std::move(pend_send.current_send));
//          pend_send.current_send = {};
        } else {
          // we need to notify about the send completion
//          send_comp_fn->sendComplete(x.second->currentSend);
//          x.second->currentSend = {};
          send_comp_fn->sendComplete(std::move(pend_send.current_send));
          // now check weather finish request is there
          if (finish_requests.find(dest) != finish_requests.end()) {
            sendFinishHeader(x);
          } else {
            pend_send.status = SEND_INIT;
          }
        }
      }
    } else if (pend_send.status == SEND_FINISH) {
//      MPI_Test(&(x.second->request), &flag, &status);
      flag = pend_send.request->waitSend();
      if (flag) {
        // LOG(INFO) << rank << " FINISHED send " << x.first;
        // we are going to send complete
//        std::shared_ptr<CylonRequest> finReq = finishRequests[x.first];
        send_comp_fn->sendFinishComplete(finish_requests[dest]);
        pend_send.status = SEND_DONE;
      }
    } else if (pend_send.status != SEND_DONE) {
      // throw an exception and log
      LOG(FATAL) << "At an un-expected state " << pend_send.status;
    }
  }
}
void GlooChannel::progressReceives() {
//  MPI_Status status;
  for (auto &x: pending_receives) {
    auto &pend_rec = x.second;
    bool flag = false;
//    status = {};
    if (pend_rec.status == RECEIVE_LENGTH_POSTED) {
//      MPI_Test(&(x.second->request), &flag, &status);
      flag = pend_rec.request->waitRecv(); // todo there could be issues here!
      if (flag) {
//        x.second->request = {};
//        int count = 0;
//        MPI_Get_count(&status, MPI_INT, &count);
        // read the length from the header
        int length = pend_rec.header_buf[0];
        int finFlag = pend_rec.header_buf[1];
        // check weather we are at the end
        if (finFlag != CYLON_MSG_FIN) {
//          if (count > 8) {
//            LOG(FATAL) << "Un-expected number of bytes expected: 8 or less "
//                       << " received: " << count;
//          }
          // malloc a buffer
          Status stat = allocator->Allocate(length, &pend_rec.data);
          if (!stat.is_ok()) {
            LOG(FATAL) << "Failed to allocate buffer with length " << length;
          }
          pend_rec.length = length;
//          MPI_Irecv(x.second->data->GetByteBuffer(), length, MPI_BYTE, x.second->receiveId, edge,
//                    comm_, &(x.second->request));
          pend_rec.request = ctx_ptr->createUnboundBuffer(pend_rec.data.get(), length);
          pend_rec.request->recv(pend_rec.recv_id, edge_, 0, length);
          pend_rec.status = RECEIVE_POSTED;
          // copy the count - 2 to the buffer
          int *header = nullptr;
//          if (count > 2) {
          header = new int[6];
          memcpy(header, pend_rec.header_buf + 2, 6 * sizeof(int));
//          }
          // notify the receiver
          rcv_fn->receivedHeader(x.first, finFlag, header, 6);
        } else {
//          if (count != 2) {
//            LOG(FATAL) << "Un-expected number of bytes expected: 2 " << " received: " << count;
//          }
          // we are not expecting to receive any more
          pend_rec.status = RECEIVED_FIN;
          // notify the receiver
          rcv_fn->receivedHeader(x.first, CYLON_MSG_FIN, nullptr, 0);
        }
      }
    } else if (pend_rec.status == RECEIVE_POSTED) {
//      MPI_Test(&x.second->request, &flag, &status);
      flag = pend_rec.request->waitRecv(); // todo there could be issues here!
      if (flag) {
        assert(x.first == pend_rec.recv_id);
//        int count = 0;
//        MPI_Get_count(&status, MPI_BYTE, &count);
//        if (count != x.second->length) {
//          LOG(FATAL) << "Un-expected number of bytes expected:" << x.second->length
//                     << " received: " << count;
//        }

//        pend_rec.request = {};
        // clear the array
        std::fill_n(pend_rec.header_buf, CYLON_CHANNEL_HEADER_SIZE, 0);
//        MPI_Irecv(x.second->headerBuf, CYLON_CHANNEL_HEADER_SIZE, MPI_INT,
//                  x.second->receiveId, edge, comm_, &(x.second->request));
        pend_rec.request = ctx_ptr->createUnboundBuffer(&pend_rec.header_buf,
                                                        CYLON_CHANNEL_HEADER_SIZE * sizeof(int));
        pend_rec.request->recv(pend_rec.recv_id, edge_, 0, CYLON_CHANNEL_HEADER_SIZE * sizeof(int));
        pend_rec.status = RECEIVE_LENGTH_POSTED;
        // call the back end
        rcv_fn->receivedData(pend_rec.recv_id, pend_rec.data, pend_rec.length);
      }
    } else if (pend_rec.status != RECEIVED_FIN) {
      LOG(FATAL) << "At an un-expected state " << pend_rec.status;
    }
  }

}
void GlooChannel::close() {
// todo close all waiting sends and receives
}
GlooChannel::GlooChannel(gloo::Context *ctx_ptr) : ctx_ptr(ctx_ptr), rank(ctx_ptr->rank) {}
}
}
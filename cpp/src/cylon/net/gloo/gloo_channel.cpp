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

GlooChannel::GlooChannel(gloo::Context *ctx_ptr) : ctx_ptr(ctx_ptr), rank(ctx_ptr->rank) {}

void GlooChannel::init(int edge,
                       const std::vector<int> &receives,
                       const std::vector<int> &sendIds,
                       ChannelReceiveCallback *rcv,
                       ChannelSendCallback *send,
                       Allocator *alloc) {
  //DLOG(INFO) << "EDGE " << edge << " RANK " << rank;
  edge_ = edge;
  rcv_fn = rcv;
  send_comp_fn = send;
  allocator = alloc;

  pending_receives.reserve(receives.size());
  sends.reserve(sendIds.size());
  finish_requests.reserve(MAX_PENDING);

  // we need to post the length buffers
  for (int source: receives) {
    // gloo can't handle local send-receives! So, local receives would be handled during sending
    if (source == rank) {
      continue;
    }

    auto iter = pending_receives.emplace(source, PendingReceive{});
    auto &p_recv = iter.first->second;
    p_recv.recv_id = source;
    if (source != rank) { // in-process receives needs to be handled separately.
      p_recv.request = ctx_ptr->createUnboundBuffer((void *) &p_recv.header_buf[0],
                                                    CYLON_CHANNEL_HEADER_SIZE * sizeof(int));
      p_recv.request->recv(source, edge);
      //DLOG(INFO) << "rcv header " << p_recv.header_buf[3] << " " << source << "->" << rank;
    } else {
      p_recv.request = {};
    }
    // set the flag to true so we can identify later which buffers are posted
    p_recv.status = RECEIVE_LENGTH_POSTED;
  }

  for (int target: sendIds) {
    sends.emplace(target, PendingSend{});
  }
}

int GlooChannel::send(std::shared_ptr<CylonRequest> request) {
  auto &pending_data = sends[request->target].pending_data;
  if (pending_data.size() > MAX_PENDING) {
    return -1;
  }
  //DLOG(INFO) << "send_buf " << request->header[1] << " " << rank << "->" << request->target;
  pending_data.push(std::move(request));
  return 1;
}

int GlooChannel::sendFin(std::shared_ptr<CylonRequest> request) {
  if (finish_requests.find(request->target) != finish_requests.end()) {
    // sendFin has already received for this target
    return -1;
  }

  //DLOG(INFO) << "send_buf fin " << request->header[1] << " " << rank << "->" << request->target;
  finish_requests.emplace(request->target, std::move(request));
  return 1;
}

void GlooChannel::sendHeader(std::pair<const int, PendingSend> &x) const {
  int dest = x.first;
  auto &pend_send = x.second;
  const auto &r = *pend_send.pending_data.front();
  assert(dest == r.target);

  // put the length to the buffer
  pend_send.header_buf[0] = r.length;
  pend_send.header_buf[1] = CYLON_MSG_NOT_FIN;

  // copy the memory of the header
  if (r.headerLength > 0) {
    memcpy(pend_send.header_buf + 2, r.header, r.headerLength * sizeof(int));
  }

  // we have to add 2 to the header length
  pend_send.request =
      ctx_ptr->createUnboundBuffer(&pend_send.header_buf[0], (2 + r.headerLength) * sizeof(int));
  pend_send.request->send(dest, edge_);
  //DLOG(INFO) << "send header " << pend_send.header_buf[3] << " " << rank << "->" << dest;
  pend_send.status = SEND_LENGTH_POSTED;
}

void GlooChannel::sendFinishHeader(std::pair<const int, PendingSend> &x) const {
  int dest = x.first;
  auto &pend_send = x.second;
  // for the last header we always send only the first 2 integers
  pend_send.header_buf[0] = 0;
  pend_send.header_buf[1] = CYLON_MSG_FIN;
  pend_send.request = ctx_ptr->createUnboundBuffer(pend_send.header_buf, 2 * sizeof(int));
  pend_send.request->send(dest, edge_);
  //DLOG(INFO) << "send finish " << pend_send.header_buf[3] << " " << rank << "->" << dest;
  pend_send.status = SEND_FINISH;
}

void GlooChannel::sendHeaderLocal(PendingSend &pend_send) {
  auto &r = *pend_send.pending_data.front();
  //DLOG(INFO) << rank << " sendHeaderLocal";
  assert(r.headerLength <= 6);
  rcv_fn->receivedHeader(rank, CYLON_MSG_NOT_FIN, r.header, r.headerLength);
  pend_send.status = SEND_LENGTH_POSTED;
}

void GlooChannel::sendFinishHeaderLocal(PendingSend &pend_send) {
  //DLOG(INFO) << rank << " sendFinishHeaderLocal";
  rcv_fn->receivedHeader(rank, CYLON_MSG_FIN, nullptr, 0);
  pend_send.status = SEND_FINISH;
}

// gloo can't handle local send-receives! So, local receives would be handled during sending
void GlooChannel::progressSendsLocal(PendingSend &pend_send) {
  if (pend_send.status == SEND_LENGTH_POSTED) {
    // now post the actual send
    // we set to the current send and pop it
    pend_send.current_send = pend_send.pending_data.front();
    const auto &r = *pend_send.current_send;
    std::shared_ptr<Buffer> data_buf;
    const auto &stat = allocator->Allocate(r.length, &data_buf);
    if (!stat.is_ok()) {
      LOG(FATAL) << "Failed to allocate buffer with length " << r.length;
    }
    std::memcpy(data_buf->GetByteBuffer(), r.buffer, r.length);
    //DLOG(INFO) << "REC_DATA_LOCAL";
    rcv_fn->receivedData(rank, std::move(data_buf), r.length);

    pend_send.pending_data.pop();
    pend_send.request = {};
    pend_send.status = SEND_POSTED;
  } else if (pend_send.status == SEND_INIT) {
    pend_send.request = {};
    // now post the actual send
    if (!pend_send.pending_data.empty()) {
      sendHeaderLocal(pend_send);
    } else if (finish_requests.find(rank) != finish_requests.end()) {
      // if there are finish requests lets send them
      sendFinishHeaderLocal(pend_send);
    }
  } else if (pend_send.status == SEND_POSTED) {
    pend_send.request = {};
    // if there are more data to post, post the length buffer now
    if (!pend_send.pending_data.empty()) {
      sendHeaderLocal(pend_send);
      // we need to notify about the send completion
      send_comp_fn->sendComplete(std::move(pend_send.current_send));
    } else {
      // we need to notify about the send completion
      send_comp_fn->sendComplete(std::move(pend_send.current_send));
      // now check weather finish request is there
      if (finish_requests.find(rank) != finish_requests.end()) {
        sendFinishHeaderLocal(pend_send);
      } else {
        pend_send.status = SEND_INIT;
      }
    }
  } else if (pend_send.status == SEND_FINISH) {
    // we are going to send complete
    send_comp_fn->sendFinishComplete(finish_requests[rank]);
    pend_send.status = SEND_DONE;
  } else if (pend_send.status != SEND_DONE) {
    // throw an exception and log
    LOG(FATAL) << "At an un-expected state " << pend_send.status;
  }
}

void GlooChannel::progressSends() {
  for (auto &x: sends) {
    int dest = x.first;
    auto &pend_send = x.second;

    if (dest == rank) { // if local, short-circuit sends
      progressSendsLocal(pend_send);
      continue;
    }

    bool flag;
    // if we are in the length posted
    if (pend_send.status == SEND_LENGTH_POSTED) {
      flag = pend_send.request->testSend();
      //DLOG(INFO) << "send header# " << pend_send.header_buf[3] << " " << rank << "->" << dest;
      if (flag) {
        pend_send.request->waitSend();
        // now post the actual send
        pend_send.current_send = pend_send.pending_data.front();

        const auto &r = *pend_send.current_send;
        assert(dest == r.target);

        pend_send.request = ctx_ptr->createUnboundBuffer(const_cast<void *>(r.buffer), r.length);
        pend_send.request->send(r.target, edge_);
        //DLOG(INFO) << "send data " << pend_send.header_buf[3] << " " << rank << "->" << dest;
        pend_send.status = SEND_POSTED;
        pend_send.pending_data.pop();
      }
    } else if (pend_send.status == SEND_INIT) {
      if (!pend_send.pending_data.empty()) {
        sendHeader(x);
      } else if (finish_requests.find(dest) != finish_requests.end()) {
        // if there are finish requests lets send them
        sendFinishHeader(x);
      }
    } else if (pend_send.status == SEND_POSTED) {
      flag = pend_send.request->testSend();
      //DLOG(INFO) << "send data# " << pend_send.header_buf[3] << " " << rank << "->" << dest;
      if (flag) {
        pend_send.request->waitSend();
        pend_send.request = {};
        // if there are more data to post, post the length buffer now
        if (!pend_send.pending_data.empty()) {
          sendHeader(x);
          // we need to notify about the send completion
          send_comp_fn->sendComplete(std::move(pend_send.current_send));
        } else {
          // we need to notify about the send completion
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
      flag = pend_send.request->testSend();
      //DLOG(INFO) << "send finish# " << pend_send.header_buf[3] << " " << rank << "->" << dest;
      if (flag) {
        pend_send.request->waitSend();
        // we are going to send complete
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
  for (auto &x: pending_receives) {
    int src = x.first;
    auto &pend_rec = x.second;

    assert(src != rank);
    assert(src == pend_rec.recv_id);

    bool flag;
    if (pend_rec.status == RECEIVE_LENGTH_POSTED) {
      //DLOG(INFO) << "rcv header$ " << src << "->" << rank;
      flag = pend_rec.request->testRecv();
      //DLOG(INFO) << "rcv header# " << pend_rec.header_buf[3] << " " << src << "->" << rank;
      if (flag) {
        pend_rec.request->waitRecv();
        // read the length from the header
        int length = pend_rec.header_buf[0];
        int finFlag = pend_rec.header_buf[1];
        // check weather we are at the end
        if (finFlag != CYLON_MSG_FIN) {
          // malloc a buffer
          Status stat = allocator->Allocate(length, &pend_rec.data);
          if (!stat.is_ok()) {
            LOG(FATAL) << "Failed to allocate buffer with length " << length;
          }
          pend_rec.length = length;
          pend_rec.request = ctx_ptr->createUnboundBuffer(pend_rec.data->GetByteBuffer(), length);
          pend_rec.request->recv(pend_rec.recv_id, edge_);
          //DLOG(INFO) << "rcv data " << pend_rec.header_buf[3] << " " << src << "->" << rank;
          pend_rec.status = RECEIVE_POSTED;
          // copy the count - 2 to the buffer
          int *header = new int[6];
          memcpy(header, &(x.second.header_buf[2]), 6 * sizeof(int));
          // notify the receiver
          rcv_fn->receivedHeader(src, CYLON_MSG_NOT_FIN, header, 6);
        } else {
          // we are not expecting to receive any more
          pend_rec.status = RECEIVED_FIN;
          // notify the receiver
          rcv_fn->receivedHeader(src, CYLON_MSG_FIN, nullptr, 0);
        }
      }
    } else if (pend_rec.status == RECEIVE_POSTED) {
      //DLOG(INFO) << "rcv data$ " << src << "->" << rank;
      flag = pend_rec.request->testRecv();
      //DLOG(INFO) << "rcv data# " << pend_rec.header_buf[3] << " " << src << "->" << rank;
      if (flag) {
        pend_rec.request->waitRecv();
        assert(src == pend_rec.recv_id);
        // clear the array
        std::fill_n(pend_rec.header_buf, CYLON_CHANNEL_HEADER_SIZE, 0);
        pend_rec.request = ctx_ptr->createUnboundBuffer(&pend_rec.header_buf,
                                                        CYLON_CHANNEL_HEADER_SIZE * sizeof(int));
        pend_rec.request->recv(pend_rec.recv_id, edge_);
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
  for (auto &p: pending_receives) {
    auto &pend_rec = p.second;
    if (pend_rec.request) {
      pend_rec.request->abortWaitRecv();
    }
  }
  pending_receives.clear();

  for (auto &p: sends) {
    auto &send = p.second;
    if (send.request) {
      send.request->abortWaitSend();
    }
  }
  sends.clear();
}

}
}
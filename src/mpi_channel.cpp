#include "mpi_channel.hpp"

#include <mpi.h>
#include <vector>
#include <iostream>

namespace twisterx {

  void MPIChannel::init(int ed, const std::vector<int>& receives, const std::vector<int>& sendIds,
                        ChannelReceiveCallback * rcv, ChannelSendCallback * send_fn) {
    edge = ed;
    rcv_fn = rcv;
    send_comp_fn = send_fn;
    // we need to post the length buffers
    for (int source : receives) {
      auto * buf = new PendingReceive();
      buf->receiveId = source;
      pendingReceives.insert(std::pair<int, PendingReceive *>(source, buf));
      MPI_Irecv(buf->headerBuf, 2, MPI_INT, source, edge, MPI_COMM_WORLD, &buf->request);
      // set the flag to true so we can identify later which buffers are posted
      buf->status = RECEIVE_LENGTH_POSTED;
    }

    for (int target : sendIds) {
      sends[target] = new PendingSend();
    }
    // get the rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "Posted receive buffers and INIT" << std::endl;
  }

  int MPIChannel::send(TxRequest * request) {
    PendingSend * ps = sends[request->target];
    if (ps->pendingData.size() > 1000) {
      return -1;
    }
    ps->pendingData.push(request);
    return 1;
  }

  int MPIChannel::sendFin(twisterx::TxRequest *request) {
    if (finishRequests.find(request->target) != finishRequests.end()) {
      return -1;
    }

    finishRequests.insert(std::pair<int, TxRequest *>(request->target, request));
    return 1;
  }

  void MPIChannel::progressReceives() {
    for (auto x : pendingReceives) {
      int flag = 0;
      MPI_Status status;

      if (x.second->status == RECEIVE_LENGTH_POSTED) {
        MPI_Test(&x.second->request, &flag, &status);
        if (flag) {
          x.second->request = {};
          // read the length from the header
          int length = x.second->headerBuf[0];
          int finFlag = x.second->headerBuf[1];
          std::cout << rank << " ** received " << length << " flag " << finFlag << std::endl;
          // check weather we are at the end
          if (finFlag != TWISTERX_MSG_FIN) {
            // malloc a buffer
            x.second->data = new char[length];
            x.second->length = length;
            MPI_Irecv(x.second->data, length, MPI_BYTE, x.second->receiveId, edge, MPI_COMM_WORLD, &x.second->request);
            x.second->status = RECEIVE_POSTED;
          } else {
            // we are not expecting to receive any more
            x.second->status = RECEIVED_FIN;
            int count = 0;
            MPI_Get_count(&status, MPI_INT, &count);
            // copy the count - 2 to the buffer

            // notify the receiver
            rcv_fn->receivedFinish(x.first);
          }
        }
      } else if (x.second->status == RECEIVE_POSTED) {
        MPI_Test(&x.second->request, &flag, &status);
        if (flag) {
          std::cout << rank << " ## received from " << x.first << " posted length receive to " << x.second->receiveId << std::endl;

          x.second->request = {};
          // clear the array
          std::fill_n(x.second->headerBuf, TWISTERX_CHANNEL_HEADER_SIZE, 0);
          // malloc a buffer
          MPI_Irecv(x.second->headerBuf, TWISTERX_CHANNEL_HEADER_SIZE, MPI_INT, x.second->receiveId, edge, MPI_COMM_WORLD, &x.second->request);
          x.second->status = RECEIVE_LENGTH_POSTED;
          // call the back end
          rcv_fn->receiveComplete(x.first, x.second->data, x.second->length);
        }
      } else {
        // we are at the end
      }
    }
  }

  void MPIChannel::progressSends() {
    // lets send values
    for (auto x : sends) {
      int flag = 0;
      MPI_Status status;
      // if we are in the length posted
      if (x.second->status == SEND_LENGTH_POSTED) {
        MPI_Test(&x.second->request, &flag, &status);
        if (flag) {
          x.second->request = {};
          // now post the actual send
          TxRequest * r = x.second->pendingData.front();
          std::cout << rank << " Sent message to " << r->target << " length " << r->length << std::endl;
          MPI_Isend(r->buffer, r->length, MPI_BYTE, r->target, edge, MPI_COMM_WORLD, &x.second->request);
          x.second->status = SEND_POSTED;
          x.second->pendingData.pop();
          // we set to the current send and pop it
          x.second->currentSend = r;
        }
      } else if (x.second->status == SEND_INIT) {
        x.second->request = {};
        // now post the actual send
        if (!x.second->pendingData.empty()) {
          sendLength(x);
        } else if (finishRequests.find(x.first) != finishRequests.end()) {
          // if there are finish requests lets send them
          sendFinishRequest(x);
        }
      } else if (x.second->status == SEND_POSTED) {
        MPI_Test(&x.second->request, &flag, &status);
        if (flag) {
          x.second->request = {};
          // if there are more data to post, post the length buffer now
          if (!x.second->pendingData.empty()) {
            sendLength(x);
            // we need to notify about the send completion
            send_comp_fn->sendComplete(x.second->currentSend);
          } else {
            // we need to notify about the send completion
            send_comp_fn->sendComplete(x.second->currentSend);
            // now check weather finish request is there
            if (finishRequests.find(x.first) != finishRequests.end()) {
              sendFinishRequest(x);
            } else {
              x.second->status = SEND_INIT;
            }
          }
        }
      } else if (x.second->status == SEND_FINISH){
        MPI_Test(&x.second->request, &flag, &status);
        if (flag) {
          std::cout << rank << " FINISHED send " << x.first << std::endl;
          // we are going to send complete
          TxRequest * finReq = finishRequests[x.first];
          send_comp_fn->sendFinishComplete(finReq);
          x.second->status = SEND_DONE;
        }
      } else {
        // throw an exception and log
//        std::cout << "ELSE " << std::endl;
      }
    }
  }

  void MPIChannel::sendLength(const std::pair<const int, PendingSend *> &x) const {
    TxRequest *r = x.second->pendingData.front();
    // put the length to the buffer
    x.second->headerBuf[0] = r->length;
    std::cout << rank << " Sent length to " << r->target << std::endl;
    MPI_Isend(&(x.second->headerBuf[0]), 2, MPI_INT, x.first, edge, MPI_COMM_WORLD, &(x.second->request));
    x.second->status = SEND_LENGTH_POSTED;
  }

  void MPIChannel::sendFinishRequest(const std::pair<const int, PendingSend *> &x) const {
    x.second->headerBuf[0] = 0;
    x.second->headerBuf[1] = TWISTERX_MSG_FIN;
    std::cout << rank << " Sent finish to " << x.first << std::endl;
    MPI_Isend(&(x.second->headerBuf[0]), 2, MPI_INT, x.first, edge, MPI_COMM_WORLD, &(x.second->request));
    x.second->status = SEND_FINISH;
  }

  void MPIChannel::close() {
    for (auto & pendingReceive : pendingReceives) {
      delete (pendingReceive.second);
    }
    pendingReceives.clear();

    for (auto & s : sends) {
      delete (s.second);
    }
    sends.clear();
  }
}
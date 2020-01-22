#include "../include/mpi_channel.hpp"

#include <mpi.h>
#include <vector>
#include <queue>

namespace twisterx {

  void MPIChannel::init(int ed, const std::vector<int>& receives, const std::vector<int>& sendIds,
                        ChannelReceiveCallback * rcv, ChannelSendCallback * send_fn) {
    edge = ed;
    rcv_fn = rcv;
    send_comp_fn = send_fn;
    // we need to post the length buffers
    for (int source : receives) {
      auto * buf = new PendingReceive();
      pendingReceives.insert(std::pair<int, PendingReceive *>(source, buf));
      MPI_Irecv(buf->headerBuf, 2, MPI_INT, source, edge, MPI_COMM_WORLD, &buf->request);
      // set the flag to true so we can identify later which buffers are posted
      buf->status = RECEIVE_LENGTH_POSTED;
    }
  }

  int MPIChannel::send(TxRequest * request) {
    PendingSend * ps = sends[request->target];
    if (ps->pendingData.size() > 1000) {
      return -1;
    }
    ps->pendingData.push(request);
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
          // malloc a buffer
          x.second->data = new char[length];
          x.second->length = length;
          MPI_Irecv(x.second->data, length, MPI_BYTE, x.second->receiveId, edge, MPI_COMM_WORLD, &x.second->request);
          x.second->status = RECEIVE_POSTED;
        }
      } else if (x.second->status == RECEIVE_POSTED) {
        MPI_Test(&x.second->request, &flag, &status);
        if (flag) {
          x.second->request = {};
          std::fill_n(x.second->headerBuf, 2, 0);
          // malloc a buffer
          MPI_Irecv(x.second->headerBuf, 2, MPI_INT, x.second->receiveId, edge, MPI_COMM_WORLD, &x.second->request);
          x.second->status = RECEIVE_LENGTH_POSTED;
          // call the back end
          rcv_fn->receiveComplete(x.first, x.second->data, x.second->length);
        }
      } else {
        // throw an exception and log
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
          MPI_Isend(r->buffer, r->length, MPI_BYTE, r->target, edge, MPI_COMM_WORLD, &x.second->request);
          x.second->status = SEND_POSTED;
          x.second->pendingData.pop();
          // we set to the current send and pop it
          x.second->currentSend = r;
          x.second->pendingData.pop();
        }
      } else if (x.second->status == SEND_INIT) {
        x.second->request = {};
        // now post the actual send
        if (!x.second->pendingData.empty()) {
          TxRequest *r = x.second->pendingData.front();
          x.second->headerBuf[0] = r->length;
          MPI_Isend(x.second->headerBuf, 2, MPI_INT, r->target, edge, MPI_COMM_WORLD, &x.second->request);
          x.second->status = SEND_LENGTH_POSTED;
        }
      } else if (x.second->status == SEND_POSTED) {
        MPI_Test(&x.second->request, &flag, &status);
        if (flag) {
          x.second->request = {};
          // if there are more data to post, post the length buffer now
          if (!x.second->pendingData.empty()) {
            TxRequest *r = x.second->pendingData.front();
            // put the length to the buffer
            x.second->headerBuf[0] = r->length;
            MPI_Isend(x.second->headerBuf, 2, MPI_INT, r->target, edge, MPI_COMM_WORLD, &x.second->request);
            x.second->status = SEND_LENGTH_POSTED;
            // we need to notify about the send completion
            send_comp_fn->sendComplete(x.second->currentSend);
          } else {
            x.second->status = SEND_INIT;
          }
        }
      } else {
        // throw an exception and log
      }
    }
  }
}
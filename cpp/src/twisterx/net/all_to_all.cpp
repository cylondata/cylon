#include <algorithm>
#include <iostream>
#include <iterator>

#include <glog/logging.h>

#include "all_to_all.hpp"
#include "mpi_channel.hpp"

namespace twisterx {
AllToAll::AllToAll(int w_id, const std::vector<int>& srcs,
                   const std::vector<int>& tgts, int edge_id, ReceiveCallback * rcvCallback) {
  worker_id = w_id;
  sources = srcs;
  targets = tgts;
  edge = edge_id;
  channel = new MPIChannel();
  channel->init(edge_id, srcs, tgts, this, this);
  callback = rcvCallback;

  // initialize the sends
  for (int t : tgts) {
    int tAdjusted = (t + w_id) % targets.size();
    sends.push_back(new AllToAllSends(tAdjusted));
  }

  thisNumTargets = 0;
  thisNumSources = 0;
  if (std::find(targets.begin(), targets.end(), w_id) != targets.end()) {
    thisNumTargets = 1;
  }

  if (std::find(sources.begin(), sources.end(), w_id) != sources.end()) {
    thisNumSources = 1;
  }
}

void AllToAll::close() {
  for (int t : targets) {
    delete sends[t];
  }
  sends.clear();
  // free the channel
  channel->close();
  delete  channel;
}

int AllToAll::insert(void *buffer, int length, int target) {
  if (finishFlag) {
    // we cannot accept further
    return -1;
  }

  AllToAllSends *s = sends[target];
  // LOG(INFO) << "Allocating buffer " << length;
  std::shared_ptr<TxRequest> request = std::make_shared<TxRequest>(target, buffer, length);
  s->requestQueue.push(request);
  s->messageSizes += length;
  return 1;
}

int AllToAll::insert(void *buffer, int length, int target, int *header, int headerLength) {
  if (finishFlag) {
    // we cannot accept further
    return -1;
  }

  // we cannot accept headers greater than 6
  if (headerLength > 6) {
    return -1;
  }

  AllToAllSends *s = sends[target];
  // LOG(INFO) << "Allocating buffer " << length;
  std::shared_ptr<TxRequest> request = std::make_shared<TxRequest>(target, buffer, length, header, headerLength);
  s->requestQueue.push(request);
  s->messageSizes += length;
  return 1;
}

bool AllToAll::isComplete() {
  bool allQueuesEmpty = true;
  // if this is a source, send until the operation is finished
  for (auto w : sends) {
    while (!w->requestQueue.empty()) {
      if (w->sendStatus == ALL_TO_ALL_FINISH_SENT || w->sendStatus == ALL_TO_ALL_FINISHED) {
        LOG(FATAL) << "We cannot have items to send after finish sent";
      }

      std::shared_ptr<TxRequest> request = w->requestQueue.front();
      // if the request is accepted to be set, pop
      if (channel->send(request)) {
        w->requestQueue.pop();
        // we add to the pending queue
        w->pendingQueue.push(request);
      }
    }

    if (w->requestQueue.empty() && w->pendingQueue.empty()) {
      if (finishFlag) {
        if (w->sendStatus == ALL_TO_ALL_SENDING) {
          std::shared_ptr<TxRequest> request = std::make_shared<TxRequest>(w->target);
          if (channel->sendFin(request)) {
            // LOG(INFO) << worker_id << " Sent FIN *** " << w.first;
            w->sendStatus = ALL_TO_ALL_FINISH_SENT;
          }
        }
      }
    } else {
      allQueuesEmpty = false;
    }
  }
  // progress the sends
  channel->progressSends();
  // progress the receives
  channel->progressReceives();

  return allQueuesEmpty && finishedTargets.size() == targets.size() && finishedSources.size() == sources.size();
}

void AllToAll::finish() {
  // here we just set the finish flag to true, the is_complete method will use this flag
  finishFlag = true;
}

void AllToAll::receivedData(int receiveId, void *buffer, int length) {
  // we just call the callback function of this
  callback->onReceive(receiveId, buffer, length);
}

void AllToAll::sendComplete(std::shared_ptr<TxRequest> request) {
  AllToAllSends *s = sends[request->target];
  s->pendingQueue.pop();
  // we sent this request so we need to reduce memory
  s->messageSizes  = s->messageSizes - request->length;
  callback->onSendComplete(request->target, request->buffer, request->length);
  // we don't have much to do here, so we delete the request
  // LOG(INFO) << worker_id << " Free buffer " << request->length;
}

void AllToAll::receivedHeader(int receiveId, int finished,
                              int * header, int headerLength) {
  if (finished) {
    // LOG(INFO) << worker_id << " Received finish " << receiveId;
    finishedSources.insert(receiveId);
    callback->onReceiveHeader(receiveId, finished, header, headerLength);
  } else {
    if (headerLength > 0) {
      // LOG(INFO) << worker_id << " Received header " << receiveId << "Header length " << headerLength;
//        for (int i = 0; i < headerLength; i++) {
//          std::cout << i << " ";
//        }
//        std::cout << std::endl;
      callback->onReceiveHeader(receiveId, finished, header, headerLength);
      delete [] header;
    } else {
      callback->onReceiveHeader(receiveId, finished, nullptr, 0);
      // LOG(INFO) << worker_id << " Received header " << receiveId << "Header length " << headerLength;
    }
  }
}

void AllToAll::sendFinishComplete(std::shared_ptr<TxRequest> request) {
  finishedTargets.insert(request->target);
  AllToAllSends *s = sends[request->target];
  s->sendStatus = ALL_TO_ALL_FINISHED;
  // LOG(INFO) << worker_id << " Free fin buffer " << request->length;
}
}

#include <algorithm>
#include "all_to_all.hpp"
#include "../include/mpi_channel.hpp"

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
      sends[t] = new AllToAllSends();
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
      sends.erase(t);
    }
  }

  int AllToAll::insert(void *buffer, int length, int target) {
    if (finishFlag) {
      // we cannot accept further
      return -1;
    }

    AllToAllSends *s = sends[target];
    // first check the size of the current buffers
    int new_length = s->messageSizes + length;
    if (new_length > 10000) {
      return 0;
    }

    auto *request = new TxRequest(target, buffer, length);
    s->requestQueue.push(request);
    s->messageSizes += length;
    return 1;
  }

  bool AllToAll::isComplete() {
    bool allQueuesEmpty = true;
    // if this is a source, send until the operation is finished
    for (auto w : sends) {
      if (!w.second->requestQueue.empty()) {
        TxRequest * request = w.second->requestQueue.front();
        // if the request is accepted to be set, pop
        if (channel->send(request)) {
          w.second->requestQueue.pop();
        }
        // if all queue are not empty set here
        if (!w.second->requestQueue.empty()) {
          allQueuesEmpty = false;
        }
      } else {
        if (finishFlag) {
          if (w.second->sendStatus == ALL_TO_ALL_SENDING) {
            auto *request = new TxRequest(w.first);
            if (channel->sendFin(request)) {
              w.second->sendStatus = ALL_TO_ALL_FINISH_SENT;
            }
          }
        }
      }
    }
    // progress the sends
    channel->progressSends();
    // progress the receives
    channel->progressReceives();

    return allQueuesEmpty && finishedTargets.size() == thisNumTargets && finishedSources.size() == thisNumTargets;
  }

  void AllToAll::finish() {
    // here we just set the finish flag to true, the is_complete method will use this flag
    finishFlag = true;
  }

  void AllToAll::receiveComplete(int receiveId, void *buffer, int length) {
    // we just call the callback function of this
    callback->onReceive(receiveId, buffer, length);
  }

  void AllToAll::sendComplete(TxRequest *request) {
    AllToAllSends *s = sends[request->target];
    // we sent this request so we need to reduce memory
    s->messageSizes  = s->messageSizes - request->length;
    // we don't have much to do here, so we delete the request
    delete request;
  }

  void AllToAll::receivedFinish(int receiveId) {
    finishedSources.insert(receiveId);
  }

  void AllToAll::sendFinishComplete(TxRequest *request) {
    finishedTargets.insert(request->target);
    delete request;
    AllToAllSends *s = sends[request->target];
    s->sendStatus = ALL_TO_ALL_FINISHED;
  }
}

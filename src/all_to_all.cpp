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
  }

  int AllToAll::insert(void *buffer, int length, int target) {
    if (finishFlag) {
      // we cannot accept further
      return -1;
    }

    // first check the size of the current buffers
    int new_length = message_sizes[target] + length;
    if (new_length > 10000) {
      return 0;
    }

    std::queue<TxRequest *> v = buffers[target];
    auto *request = new TxRequest(target, buffer, length);
    v.push(request);
    message_sizes.insert(std::pair<int, int>(target, length));
    return 1;
  }

  bool AllToAll::isComplete() {
    // if this is a source, send until the operation is finished
    for (auto w : buffers) {
      if (!w.second.empty()) {
        TxRequest * request = w.second.front();
        // if the request is accepted to be set, pop
        if (channel->send(request)) {
          w.second.pop();
        }
      }
    }
    // progress the sends
    channel->progressSends();
    // progress the receives
    channel->progressReceives();
    // if this is a
    return false;
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
    // we don't have much to do here, so we delete the request
    delete request;
  }
}

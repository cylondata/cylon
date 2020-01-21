#include "all_to_all.hpp"

namespace twisterx {

  AllToAll::AllToAll(int w_id, std::vector<int> srcs,
      std::vector<int> tgts, int edge_id) {
    worker_id = w_id;
    sources = srcs;
    targets = tgts;
    edge = edge_id;
  }

  bool AllToAll::insert(void *buffer, int length, int target) {
    // first check the size of the current buffers
    int new_length = message_sizes[target] + length;
    if (new_length > 10000) {
      return false;
    }

    std::vector<void *> v = buffers[target];
    v.push_back(buffer);
    message_sizes.insert(std::pair<int, int>(target, length));
    return true;
  }

  bool AllToAll::is_complete() {

    return false;
  }
}

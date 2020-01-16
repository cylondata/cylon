#include "all_to_all.hpp"

namespace twisterx {

  AllToAll::AllTlAll(int w_id, std::vector<int> workers) {
    worker_id = w_id;
    all_workers = workers;
  }

  bool AllToAll::insert(void *buffer, int length, int target) {

  }

  bool AllToAll::is_complete() {

  }
}

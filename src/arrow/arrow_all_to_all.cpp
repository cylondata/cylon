//
// Created by skamburu on 1/29/20.
//

#include "arrow_all_to_all.hpp"

namespace twisterx {
  ArrowAllToAll::ArrowAllToAll(int worker_id, const std::vector<int> &source, const std::vector<int> &targets,
                               int edgeId, twisterx::ReceiveCallback *callback) {

  }

  int ArrowAllToAll::insert(arrow::Table *arrow, int length, int target) {
    
  }
}
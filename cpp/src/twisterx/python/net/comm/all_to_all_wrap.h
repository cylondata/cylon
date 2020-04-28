//
// Created by vibhatha on 4/26/20.
//

#ifndef TWISTERX_SRC_TWISTERX_PYTHON_NET_COMM_ALL_TO_ALL_WRAP_H_
#define TWISTERX_SRC_TWISTERX_PYTHON_NET_COMM_ALL_TO_ALL_WRAP_H_

#include "../../../net/all_to_all.hpp"
#include "Callback.h"

using namespace twisterx;

namespace twisterx {
namespace net {
namespace comm {
class all_to_all_wrap{
 private:
  std::vector<int> sources = {0};
  std::vector<int> targets = {0};
  twisterx::net::comms::Callback callback_;
  twisterx::AllToAll *all_;
 public:
  all_to_all_wrap();
  all_to_all_wrap(int worker_id, const std::vector<int> &source, const std::vector<int> &targets, int edgeId);
  void insert(void *buffer, int length, int target, int *header, int headerLength);
  int insert(void *buffer, int length, int target);
  void wait();
  void finish();
  void set_instance(twisterx::AllToAll *all);
  twisterx::AllToAll* get_instance();
  void init_all_to_all(int worker_id, const std::vector<int> &source, const std::vector<int> &targets, int edgeId, void *buffer, int length, int target, int *header, int headerLength);
  void execute();
};
}
}
}

#endif //TWISTERX_SRC_TWISTERX_PYTHON_NET_COMM_ALL_TO_ALL_WRAP_H_

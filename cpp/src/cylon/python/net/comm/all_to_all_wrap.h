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

#ifndef CYLON_SRC_CYLON_PYTHON_NET_COMM_ALL_TO_ALL_WRAP_H_
#define CYLON_SRC_CYLON_PYTHON_NET_COMM_ALL_TO_ALL_WRAP_H_

#include "net/ops/all_to_all.hpp"
#include "net/mpi/mpi_communicator.hpp"
#include "callback.h"

using namespace cylon;

namespace cylon {
namespace net {
namespace comm {
class all_to_all_wrap {
 private:
  std::vector<int> sources = {0};
  std::vector<int> targets = {0};
  cylon::net::comms::Callback callback_;
  cylon::AllToAll *all_;
 public:
  all_to_all_wrap();
  all_to_all_wrap(int worker_id, const std::vector<int> &source, const std::vector<int> &targets, int edgeId);
  void insert(void *buffer, int length, int target, int *header, int headerLength);
  int insert(void *buffer, int length, int target);
  void wait();
  void finish();
  void set_instance(cylon::AllToAll *all);
  cylon::AllToAll *get_instance();
};
}
}
}

#endif //CYLON_SRC_CYLON_PYTHON_NET_COMM_ALL_TO_ALL_WRAP_H_

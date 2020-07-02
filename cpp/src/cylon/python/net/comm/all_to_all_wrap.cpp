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

#include "all_to_all_wrap.h"
#include "callback.h"

using namespace cylon::net::comms;

namespace cylon {
namespace net {
namespace comm {

cylon::net::comm::all_to_all_wrap::all_to_all_wrap() {

}

cylon::net::comm::all_to_all_wrap::all_to_all_wrap(int worker_id,
													  const std::vector<int> &source,
													  const std::vector<int> &targets,
													  int edgeId) {
  Callback *callback = new Callback();
  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
  all_ = new cylon::AllToAll(ctx, source, targets, edgeId, callback);

}

void cylon::net::comm::all_to_all_wrap::set_instance(cylon::AllToAll *all) {
  all_ = all;
}

void cylon::net::comm::all_to_all_wrap::insert(void *buffer, int length, int target, int *header, int headerLength) {
  this->all_->insert(buffer, length, target, header, headerLength);
}

int cylon::net::comm::all_to_all_wrap::insert(void *buffer, int length, int target) {
  all_->insert(buffer, length, target);
}

cylon::AllToAll *cylon::net::comm::all_to_all_wrap::get_instance() {
  return all_;
}

void cylon::net::comm::all_to_all_wrap::wait() {
  this->all_->finish();
  while (this->all_->isComplete()) {

  }
}

void cylon::net::comm::all_to_all_wrap::finish() {
  all_->close();
}

}
}
}
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

using namespace twisterx::net::comms;

namespace twisterx {
namespace net {
namespace comm {

twisterx::net::comm::all_to_all_wrap::all_to_all_wrap() {

}

twisterx::net::comm::all_to_all_wrap::all_to_all_wrap(int worker_id,
													  const std::vector<int> &source,
													  const std::vector<int> &targets,
													  int edgeId) {
  Callback *callback = new Callback();
  auto mpi_config = new twisterx::net::MPIConfig();
  auto ctx = twisterx::TwisterXContext::InitDistributed(mpi_config);
  all_ = new twisterx::AllToAll(ctx, source, targets, edgeId, callback);

}

void twisterx::net::comm::all_to_all_wrap::set_instance(twisterx::AllToAll *all) {
  all_ = all;
}

void twisterx::net::comm::all_to_all_wrap::insert(void *buffer, int length, int target, int *header, int headerLength) {
  this->all_->insert(buffer, length, target, header, headerLength);
}

int twisterx::net::comm::all_to_all_wrap::insert(void *buffer, int length, int target) {
  all_->insert(buffer, length, target);
}

twisterx::AllToAll *twisterx::net::comm::all_to_all_wrap::get_instance() {
  return all_;
}

void twisterx::net::comm::all_to_all_wrap::wait() {
  this->all_->finish();
  while (this->all_->isComplete()) {

  }
}

void twisterx::net::comm::all_to_all_wrap::finish() {
  all_->close();
}

}
}
}
//
// Created by vibhatha on 4/26/20.
//

#include "all_to_all_wrap.h"
#include "Callback.h"

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

twisterx::AllToAll * twisterx::net::comm::all_to_all_wrap::get_instance() {
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
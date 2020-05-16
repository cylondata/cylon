//
// Created by vibhatha on 5/15/20.
//

#include "twisterx_context_wrap.h"
#include "../../net/mpi/mpi_communicator.h"

TwisterXContext * twisterx::py::twisterx_context_wrap::Init() {
  new TwisterXContext(false);
}

TwisterXContext * twisterx::py::twisterx_context_wrap::InitDistributed(std::string config) {
  if (config == "mpi") {
	auto ctx = new TwisterXContext(true);
	ctx->setCommunicator(new net::MPICommunicator());
	auto mpi_config = new twisterx::net::MPIConfig();
	ctx->GetCommunicator()->Init(mpi_config);
	ctx->setDistributed(true);
	return ctx;
  } else {
	throw "Unsupported communication type";
  }
  return nullptr;
}

net::Communicator * twisterx::py::twisterx_context_wrap::GetCommunicator() const {
  return this->communicator;
}

void twisterx::py::twisterx_context_wrap::AddConfig(const std::string &key, const std::string &value) {
  this->config.insert(std::pair<std::string, std::string>(key, value));
}




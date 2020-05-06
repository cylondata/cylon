#include "twisterx_context.h"
#include "../net/mpi/mpi_communicator.h"

namespace twisterx {

TwisterXContext *TwisterXContext::Init() {
  return new TwisterXContext(false);
}
TwisterXContext::TwisterXContext(bool distributed) {
  this->distributed = distributed;
}

TwisterXContext *TwisterXContext::InitDistributed(net::CommConfig *config) {
  if (config->Type() == net::CommType::MPI) {
	auto ctx = new TwisterXContext(true);
	ctx->communicator = new net::MPICommunicator();
	ctx->communicator->Init(config);
	return ctx;
  } else {
	throw "Unsupported communication type";
  }
  return nullptr;
}
net::Communicator *TwisterXContext::GetCommunicator() const {
  return this->communicator;
}
void TwisterXContext::AddConfig(const std::string &key, const std::string &value) {
  this->config.insert(std::pair<std::string, std::string>(key, value));
}
std::string TwisterXContext::GetConfig(const std::string &key, const std::string &def) {
  auto find = this->config.find(key);
  if (find == this->config.end()) {
	return def;
  }
  return find->second;
}
int TwisterXContext::GetRank() {
  if (this->distributed) {
	this->communicator->GetRank();
  }
  return 0;
}
int TwisterXContext::GetWorldSize() {
  if (this->distributed) {
	this->communicator->GetWorldSize();
  }
  return 1;
}
void TwisterXContext::Finalize() {
  if (this->distributed) {
	delete this->communicator;
  }
}
}
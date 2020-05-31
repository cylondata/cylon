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

#include "twisterx_context_wrap.h"
#include "../../net/mpi/mpi_communicator.h"

//TwisterXContext * twisterx::py::twisterx_context_wrap::Init() {
//  new TwisterXContext(false);
//}
//
//TwisterXContext * twisterx::py::twisterx_context_wrap::InitDistributed(std::string config) {
//  if (config == "mpi") {
//	auto ctx = new TwisterXContext(true);
//	ctx->setCommunicator(new net::MPICommunicator());
//	auto mpi_config = new twisterx::net::MPIConfig();
//	ctx->GetCommunicator()->Init(mpi_config);
//	ctx->setDistributed(true);
//	return ctx;
//  } else {
//	throw "Unsupported communication type";
//  }
//  return nullptr;
//}

twisterx::py::twisterx_context_wrap::twisterx_context_wrap() {
  this->context = new TwisterXContext(false);
  this->distributed = false;
}

twisterx::py::twisterx_context_wrap::twisterx_context_wrap(std::string config) {
  if (config == "mpi") {
	auto ctx = new TwisterXContext(true);
	this->distributed = true;
	ctx->setCommunicator(new net::MPICommunicator());
	auto mpi_config = new twisterx::net::MPIConfig();
	ctx->GetCommunicator()->Init(mpi_config);
	ctx->setDistributed(true);
	this->context = ctx;
  } else {
	throw "Unsupported communication type";
  }
}

TwisterXContext * twisterx::py::twisterx_context_wrap::getInstance() {
  return context;
}


net::Communicator * twisterx::py::twisterx_context_wrap::GetCommunicator() const {
  return this->context->GetCommunicator();
}

void twisterx::py::twisterx_context_wrap::AddConfig(const std::string &key, const std::string &value) {
  this->config.insert(std::pair<std::string, std::string>(key, value));
}

std::string twisterx::py::twisterx_context_wrap::GetConfig(const std::string &key, const std::string &defn) {
  auto find = this->config.find(key);
  if (find == this->config.end()) {
	return defn;
  }
  return find->second;
}

int twisterx::py::twisterx_context_wrap::GetRank() {
  if (this->distributed) {
	return this->context->GetCommunicator()->GetRank();
  }
  return 0;
}

int twisterx::py::twisterx_context_wrap::GetWorldSize() {
  if (this->distributed) {
	return this->context->GetCommunicator()->GetWorldSize();
  }
  return 1;
}

void twisterx::py::twisterx_context_wrap::Finalize() {
  if (this->distributed) {
	this->context->GetCommunicator()->Finalize();
	delete this->context->GetCommunicator();
  }
}

vector<int> twisterx::py::twisterx_context_wrap::GetNeighbours(bool include_self) {
  vector<int> neighbours{};
  neighbours.reserve(this->GetWorldSize());
  for (int i = 0; i < this->GetWorldSize(); i++) {
	if (i == this->GetRank() && !include_self) {
	  continue;
	}
	neighbours.push_back(i);
  }
  return neighbours;
}





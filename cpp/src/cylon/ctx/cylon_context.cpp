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

#include "cylon_context.hpp"

#include <utility>
#include <vector>

#include "arrow/memory_pool.h"
#include "../net/mpi/mpi_communicator.hpp"

namespace cylon {

CylonContext *CylonContext::Init() {
  return new CylonContext(false);
}
CylonContext::CylonContext(bool distributed) {
  this->distributed = distributed;
}

CylonContext *CylonContext::InitDistributed(net::CommConfig *config) {
  if (config->Type() == net::CommType::MPI) {
    auto ctx = new CylonContext(true);
    ctx->communicator = new net::MPICommunicator();
    ctx->communicator->Init(config);
    ctx->distributed = true;
    return ctx;
  } else {
    throw "Unsupported communication type";
  }
  return nullptr;
}
net::Communicator *CylonContext::GetCommunicator() const {
  return this->communicator;
}

void CylonContext::setCommunicator(net::Communicator *communicator1) {
  this->communicator = communicator1;
}

void CylonContext::setDistributed(bool distributed) {
  this->distributed = distributed;
}

void CylonContext::AddConfig(const std::string &key, const std::string &value) {
  this->config.insert(std::pair<std::string, std::string>(key, value));
}
std::string CylonContext::GetConfig(const std::string &key, const std::string &def) {
  auto find = this->config.find(key);
  if (find == this->config.end()) {
    return def;
  }
  return find->second;
}
int CylonContext::GetRank() {
  if (this->distributed) {
    return this->communicator->GetRank();
  }
  return 0;
}
int CylonContext::GetWorldSize() {
  if (this->distributed) {
    return this->communicator->GetWorldSize();
  }
  return 1;
}
void CylonContext::Finalize() {
  if (this->distributed) {
    this->communicator->Finalize();
  }
}
vector<int> CylonContext::GetNeighbours(bool include_self) {
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

cylon::MemoryPool *CylonContext::GetMemoryPool() {
  return this->memory_pool;
}

void CylonContext::SetMemoryPool(cylon::MemoryPool *mem_pool) {
  this->memory_pool = mem_pool;
}
int32_t CylonContext::GetNextSequence() {
  return this->sequence_no++;
}
}  // namespace cylon

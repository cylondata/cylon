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

#include "twisterx_context.h"
#include "arrow/memory_pool.h"
#include "../net/mpi/mpi_communicator.h"

namespace twisterx {

arrow::Status ArrowStatus(twisterx::Status status) {
  return arrow::Status(static_cast<arrow::StatusCode>(status.get_code()), status.get_msg());
};

class ProxyMemoryPool : public arrow::MemoryPool {

 private:
  twisterx::MemoryPool *tx_memory;
 public:
  ProxyMemoryPool(twisterx::MemoryPool *tx_memory) {
    this->tx_memory = tx_memory;
  }

  ~ProxyMemoryPool() override {
    delete tx_memory;
  }

  arrow::Status Allocate(int64_t size, uint8_t **out) override {
    return ArrowStatus(tx_memory->Allocate(size, out));
  }

  arrow::Status Reallocate(int64_t old_size, int64_t new_size, uint8_t **ptr) override {
    return ArrowStatus(tx_memory->Reallocate(old_size, new_size, ptr));
  };

  void Free(uint8_t *buffer, int64_t size) override {
    tx_memory->Free(buffer, size);
  }

  int64_t bytes_allocated() const override {
    return this->tx_memory->bytes_allocated();
  }

  int64_t max_memory() const override {
    return this->tx_memory->max_memory();
  }

  std::string backend_name() const override {
    return this->tx_memory->backend_name();
  }
};

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
    ctx->distributed = true;
    return ctx;
  } else {
    throw "Unsupported communication type";
  }
  return nullptr;
}
net::Communicator *TwisterXContext::GetCommunicator() const {
  return this->communicator;
}

void TwisterXContext::setCommunicator(net::Communicator *communicator1) {
  this->communicator = communicator1;
}

void TwisterXContext::setDistributed(bool distributed) {
  this->distributed = distributed;
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
    return this->communicator->GetRank();
  }
  return 0;
}
int TwisterXContext::GetWorldSize() {
  if (this->distributed) {
    return this->communicator->GetWorldSize();
  }
  return 1;
}
void TwisterXContext::Finalize() {
  if (this->distributed) {
    this->communicator->Finalize();
    delete this->communicator;
  }
}
vector<int> TwisterXContext::GetNeighbours(bool include_self) {
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

template<typename TYPE>
TYPE *TwisterXContext::GetMemoryPool() {
  if (this->memory_pool == nullptr) {
    return arrow::default_memory_pool();
  } else {
    return new ProxyMemoryPool(this->memory_pool);
  }
}

void TwisterXContext::SetMemoryPool(twisterx::MemoryPool *mem_pool) {
  this->memory_pool = mem_pool;
}
}
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

#ifndef CYLON_SRC_CYLON_CTX_ARROW_MEMORY_POOL_UTILS_HPP_
#define CYLON_SRC_CYLON_CTX_ARROW_MEMORY_POOL_UTILS_HPP_

#include "arrow//memory_pool.h"
#include "cylon_context.hpp"

namespace cylon {

arrow::Status ArrowStatus(cylon::Status status);

class ProxyMemoryPool : public arrow::MemoryPool {

 private:
  cylon::MemoryPool *tx_memory;
 public:
  explicit ProxyMemoryPool(cylon::MemoryPool *tx_memory) {
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

arrow::MemoryPool *ToArrowPool(shared_ptr<cylon::CylonContext> &ctx);
}

#endif //CYLON_SRC_CYLON_CTX_ARROW_MEMORY_POOL_UTILS_HPP_

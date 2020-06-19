#ifndef TWISTERX_SRC_TWISTERX_CTX_ARROW_MEMORY_POOL_UTILS_H_
#define TWISTERX_SRC_TWISTERX_CTX_ARROW_MEMORY_POOL_UTILS_H_

#include "arrow//memory_pool.h"
#include "twisterx_context.h"

namespace twisterx {

arrow::Status ArrowStatus(twisterx::Status status);

class ProxyMemoryPool : public arrow::MemoryPool {

 private:
  twisterx::MemoryPool *tx_memory;
 public:
  explicit ProxyMemoryPool(twisterx::MemoryPool *tx_memory) {
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

arrow::MemoryPool *ToArrowPool(twisterx::TwisterXContext *ctx);
}

#endif //TWISTERX_SRC_TWISTERX_CTX_ARROW_MEMORY_POOL_UTILS_H_

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


#ifndef CYLON_CPP_SRC_CYLON_ARROW_ARROW_BUFFER_HPP_
#define CYLON_CPP_SRC_CYLON_ARROW_ARROW_BUFFER_HPP_

#include <arrow/api.h>

#include "cylon/net/buffer.hpp"

namespace cylon {

/**
 * Arrow table specific buffer
 */
class ArrowBuffer : public Buffer {
 public:
  explicit ArrowBuffer(std::shared_ptr<arrow::Buffer> buf);
  int64_t GetLength() const override;
  uint8_t *GetByteBuffer() override;

  const std::shared_ptr<arrow::Buffer> &getBuf() const;
 private:
  std::shared_ptr<arrow::Buffer> buf;
};

/**
 * Arrow table specific allocator
 */
class ArrowAllocator : public Allocator {
 public:
  explicit ArrowAllocator(arrow::MemoryPool *pool = arrow::default_memory_pool());
  ~ArrowAllocator() override;

  Status Allocate(int64_t length, std::shared_ptr<Buffer> *buffer) override;
 private:
  arrow::MemoryPool *pool;
};

}

#endif //CYLON_CPP_SRC_CYLON_ARROW_ARROW_BUFFER_HPP_

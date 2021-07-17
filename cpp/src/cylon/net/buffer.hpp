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
#ifndef CYLON_BUFFER_H
#define CYLON_BUFFER_H

#include <memory>

#include <cylon/status.hpp>

namespace cylon {
  /**
   * Represent a buffer abstraction for the channel
   */
  class Buffer {
  public:
    virtual ~Buffer() = default;
    virtual int64_t GetLength() = 0;
    virtual uint8_t * GetByteBuffer() = 0;
  };

  /**
   * A custom allocator for the channel
   */
  class Allocator {
  public:
    virtual Status Allocate(int64_t length, std::shared_ptr<Buffer> *buffer) = 0;
  };

  class DefaultBuffer : public Buffer {
   public:
    int64_t GetLength() override {
      return length;
    }
    uint8_t * GetByteBuffer() override {
      return buf;
    }
    DefaultBuffer(uint8_t *buf, int64_t length) : buf(buf), length(length) {}
   private:
    uint8_t *buf;
    int64_t length;
  };

  class DefaultAllocator : public Allocator {
   public:
    cylon::Status Allocate(int64_t length,
        std::shared_ptr<Buffer> *buffer) override {
      auto *b = new uint8_t[length];
      *buffer = std::make_shared<DefaultBuffer>(b, length);
      return Status::OK();
    }
  };
}  // namespace cylon

#endif //CYLON_BUFFER_H

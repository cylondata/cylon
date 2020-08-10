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

#include "status.hpp"

namespace cylon {
  class Buffer {
  public:
    virtual int64_t GetLength() = 0;
    virtual uint8_t * GetByteBuffer() = 0;
  };

  class Allocator {
  public:
    virtual Status Allocate(int64_t length, std::shared_ptr<Buffer> *buffer) = 0;
  };
}  // namespace cylon

#endif //CYLON_BUFFER_H

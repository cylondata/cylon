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

#include "cylon/arrow/arrow_buffer.hpp"
#include "cylon/util/macros.hpp"

namespace cylon {

ArrowAllocator::ArrowAllocator(arrow::MemoryPool *pool) : pool(pool) {}

ArrowAllocator::~ArrowAllocator() = default;

Status ArrowAllocator::Allocate(int64_t length, std::shared_ptr<Buffer> *buffer) {
  CYLON_ASSIGN_OR_RAISE(auto alloc_buf, arrow::AllocateBuffer(length, pool))
  *buffer = std::make_shared<ArrowBuffer>(std::move(alloc_buf));
  return Status::OK();
}

int64_t ArrowBuffer::GetLength() const {
  return 0;
}

uint8_t *ArrowBuffer::GetByteBuffer() {
  return buf->mutable_data();
}

ArrowBuffer::ArrowBuffer(std::shared_ptr<arrow::Buffer> buf) : buf(std::move(buf)) {}

const std::shared_ptr<arrow::Buffer> &ArrowBuffer::getBuf() const {
  return buf;
}


}

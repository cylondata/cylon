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
#ifndef GCYLON_CUDF_BUFFER_H
#define GCYLON_CUDF_BUFFER_H

#include <cylon/status.hpp>
#include <cylon/net/buffer.hpp>
#include <rmm/device_buffer.hpp>

namespace gcylon {

class CudfBuffer : public cylon::Buffer {
public:
  CudfBuffer(std::shared_ptr<rmm::device_buffer> rmm_buf);
  int64_t GetLength() const override;
  uint8_t * GetByteBuffer() override;
  const std::shared_ptr<rmm::device_buffer>& getBuf() const;
private:
  std::shared_ptr<rmm::device_buffer> rmm_buf;
};

class CudfAllocator : public cylon::Allocator {
public:
  cylon::Status Allocate(int64_t length, std::shared_ptr<cylon::Buffer> *buffer) override;
  virtual ~CudfAllocator();
};

}// end of namespace gcylon

#endif //GCYLON_CUDF_BUFFER_H

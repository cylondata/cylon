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

#include <glog/logging.h>
#include <gcylon/cudf_buffer.hpp>

namespace gcylon {

//////////////////////////////////////////////////////////////////////
// CudfBuffer implementations
//////////////////////////////////////////////////////////////////////
    CudfBuffer::CudfBuffer(std::shared_ptr<rmm::device_buffer> rmm_buf) : rmm_buf(std::move(rmm_buf)) {}

    int64_t CudfBuffer::GetLength() const {
        return rmm_buf->size();
    }

    uint8_t * CudfBuffer::GetByteBuffer() {
        return (uint8_t *)rmm_buf->data();
    }

    const std::shared_ptr<rmm::device_buffer>& CudfBuffer::getBuf() const {
        return rmm_buf;
    }

//////////////////////////////////////////////////////////////////////
// CudfAllocator implementations
//////////////////////////////////////////////////////////////////////
    cylon::Status CudfAllocator::Allocate(int64_t length, std::shared_ptr<cylon::Buffer> *buffer) {
        try {
            auto rmm_buf = std::make_shared<rmm::device_buffer>(length, rmm::cuda_stream_default);
            *buffer = std::make_shared<CudfBuffer>(std::move(rmm_buf));
            return cylon::Status::OK();
        } catch (rmm::bad_alloc * badAlloc) {
            LOG(ERROR) << "failed to allocate gpu memory with rmm: " << badAlloc->what();
            return cylon::Status(cylon::Code::GpuMemoryError);
        }
    }

    CudfAllocator::~CudfAllocator() = default;

}// end of namespace gcylon

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

#ifndef CYLON_CPP_SRC_CYLON_NET_OPS_BASE_OPS_HPP_
#define CYLON_CPP_SRC_CYLON_NET_OPS_BASE_OPS_HPP_

#include "cylon/net/serialize.hpp"
#include "cylon/net/buffer.hpp"

namespace cylon {
namespace net {

class TableAllgatherImpl {
 public:
  explicit TableAllgatherImpl(const int num_buffers) : num_buffers_(num_buffers) {};

  virtual ~TableAllgatherImpl() = default;

  virtual Status AllgatherBufferSizes(const int32_t *send_data, int32_t *rcv_data) = 0;

  virtual Status IallgatherBufferData(int buf_idx,
                                      const uint8_t *send_data,
                                      int32_t send_count,
                                      uint8_t *recv_data,
                                      const std::vector<int32_t> &recv_count,
                                      const std::vector<int32_t> &displacements) = 0;

  virtual Status WaitAll() = 0;

  Status Execute(const std::shared_ptr<cylon::TableSerializer> &serializer,
                 const std::shared_ptr<cylon::Allocator> &allocator,
                 std::vector<int32_t> &all_buffer_sizes,
                 std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                 std::vector<std::vector<int32_t>> &displacements,
                 int world_size);

 protected:
  const int num_buffers_;
};

class TableGatherImpl {
 public:
  explicit TableGatherImpl(const int num_buffers) : num_buffers_(num_buffers) {}
  virtual ~TableGatherImpl() = default;

  virtual cylon::Status GatherBufferSizes(const int32_t *send_data,
                                          int32_t *rcv_data,
                                          int gather_root) = 0;

  virtual Status IgatherBufferData(int buf_idx,
                                   const uint8_t *send_data,
                                   int32_t send_count,
                                   uint8_t *recv_data,
                                   const std::vector<int32_t> &recv_count,
                                   const std::vector<int32_t> &displacements,
                                   int gather_root) = 0;

  virtual Status WaitAll() = 0;

  Status Execute(const std::shared_ptr<cylon::TableSerializer> &serializer,
                 int gather_root,
                 bool gather_from_root,
                 const std::shared_ptr<cylon::Allocator> &allocator,
                 std::vector<int32_t> &all_buffer_sizes,
                 std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                 std::vector<std::vector<int32_t>> &displacements,
                 int rank,
                 int world_size);

 protected:
  const int num_buffers_;
};

}
}

#endif //CYLON_CPP_SRC_CYLON_NET_OPS_BASE_OPS_HPP_

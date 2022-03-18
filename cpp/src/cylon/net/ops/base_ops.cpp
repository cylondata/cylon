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

#include "base_ops.hpp"
#include "cylon/util/macros.hpp"
#include "cylon/net/utils.hpp"

namespace cylon {
namespace net {

Status TableAllgatherImpl::Execute(const std::shared_ptr<cylon::TableSerializer> &serializer,
                                   const std::shared_ptr<cylon::Allocator> &allocator,
                                   std::vector<int32_t> &all_buffer_sizes,
                                   std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                                   std::vector<std::vector<int32_t>> &displacements,
                                   int world_size) {
  // first gather table buffer sizes
  const auto &local_buffer_sizes = serializer->getBufferSizes();

  all_buffer_sizes.resize(world_size * num_buffers_);

  RETURN_CYLON_STATUS_IF_FAILED(AllgatherBufferSizes(local_buffer_sizes.data(),
                                                     all_buffer_sizes.data()));

  const auto &total_buffer_sizes = totalBufferSizes(all_buffer_sizes, num_buffers_, world_size);

  const std::vector<const uint8_t *> &send_buffers = serializer->getDataBuffers();

  for (int32_t i = 0; i < num_buffers_; ++i) {
    std::shared_ptr<cylon::Buffer> receive_buf;
    RETURN_CYLON_STATUS_IF_FAILED(allocator->Allocate(total_buffer_sizes[i], &receive_buf));
    const auto &receive_counts = receiveCounts(all_buffer_sizes, i, num_buffers_,
                                               world_size);
    auto disp_per_buffer = displacementsPerBuffer(all_buffer_sizes, i, num_buffers_,
                                                  world_size);

    RETURN_CYLON_STATUS_IF_FAILED(IallgatherBufferData(i,
                                                       send_buffers[i],
                                                       local_buffer_sizes[i],
                                                       receive_buf->GetByteBuffer(),
                                                       receive_counts,
                                                       disp_per_buffer));
    displacements.push_back(std::move(disp_per_buffer));
    received_buffers.push_back(std::move(receive_buf));
  }

  return WaitAll();
}

Status TableGatherImpl::Execute(const std::shared_ptr<cylon::TableSerializer> &serializer,
                                int gather_root,
                                bool gather_from_root,
                                const std::shared_ptr<cylon::Allocator> &allocator,
                                std::vector<int32_t> &all_buffer_sizes,
                                std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                                std::vector<std::vector<int32_t>> &displacements,
                                int rank,
                                int world_size) {
  bool is_root = gather_root == rank;
  // first gather table buffer sizes
  std::vector<int32_t> local_buffer_sizes;
  if (is_root && !gather_from_root) {
    local_buffer_sizes = serializer->getEmptyTableBufferSizes();
  } else {
    local_buffer_sizes = serializer->getBufferSizes();
  }

  int32_t num_buffers = local_buffer_sizes.size();

  // gather size buffers
  if (is_root) {
    all_buffer_sizes.resize(world_size * num_buffers);
  }

  RETURN_CYLON_STATUS_IF_FAILED(GatherBufferSizes(local_buffer_sizes.data(),
                                                  all_buffer_sizes.data(), gather_root));

  std::vector<int32_t> total_buffer_sizes;
  if (is_root) {
    totalBufferSizes(all_buffer_sizes, num_buffers, world_size).swap(total_buffer_sizes);
  }

  const std::vector<const uint8_t *> &send_buffers = serializer->getDataBuffers();

  for (int32_t i = 0; i < num_buffers; ++i) {
    if (is_root) {
      std::shared_ptr<cylon::Buffer> receive_buf;
      RETURN_CYLON_STATUS_IF_FAILED(allocator->Allocate(total_buffer_sizes[i], &receive_buf));
      const auto &receive_counts = receiveCounts(all_buffer_sizes, i,
                                                 num_buffers, world_size);
      auto disp_per_buffer = displacementsPerBuffer(all_buffer_sizes, i,
                                                    num_buffers, world_size);

      RETURN_CYLON_STATUS_IF_FAILED(IgatherBufferData(i,
                                                      send_buffers[i],
                                                      local_buffer_sizes[i],
                                                      receive_buf->GetByteBuffer(),
                                                      receive_counts,
                                                      disp_per_buffer,
                                                      gather_root));
      displacements.push_back(std::move(disp_per_buffer));
      received_buffers.push_back(std::move(receive_buf));
    } else {
      RETURN_CYLON_STATUS_IF_FAILED(IgatherBufferData(i,
                                                      send_buffers[i],
                                                      local_buffer_sizes[i],
                                                      nullptr,
                                                      {},
                                                      {},
                                                      gather_root));
    }
  }

  return WaitAll();
}

}// net
}// cylon


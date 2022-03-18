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
#include "cylon/serialize/table_serialize.hpp"
#include "cylon/arrow/arrow_buffer.hpp"

namespace cylon {
namespace net {

Status TableAllgatherImpl::Execute(const std::shared_ptr<TableSerializer> &serializer,
                                   const std::shared_ptr<cylon::Allocator> &allocator,
                                   int world_size,
                                   std::vector<int32_t> *all_buffer_sizes,
                                   std::vector<std::shared_ptr<Buffer>> *received_buffers,
                                   std::vector<std::vector<int32_t>> *displacements) {
  int num_buffers = serializer->getNumberOfBuffers();

  // initialize the impl
  Init(num_buffers);

  // first gather table buffer sizes
  const auto &local_buffer_sizes = serializer->getBufferSizes();

  all_buffer_sizes->resize(world_size * num_buffers);

  RETURN_CYLON_STATUS_IF_FAILED(AllgatherBufferSizes(local_buffer_sizes.data(), num_buffers,
                                                     all_buffer_sizes->data()));

  const auto &total_buffer_sizes = totalBufferSizes(*all_buffer_sizes, num_buffers, world_size);

  const std::vector<const uint8_t *> &send_buffers = serializer->getDataBuffers();

  displacements->reserve(num_buffers);
  received_buffers->reserve(num_buffers);
  for (int32_t i = 0; i < num_buffers; ++i) {
    std::shared_ptr<cylon::Buffer> receive_buf;
    RETURN_CYLON_STATUS_IF_FAILED(allocator->Allocate(total_buffer_sizes[i], &receive_buf));
    const auto &receive_counts = receiveCounts(*all_buffer_sizes, i, num_buffers,
                                               world_size);
    auto disp_per_buffer = displacementsPerBuffer(*all_buffer_sizes, i, num_buffers,
                                                  world_size);

    RETURN_CYLON_STATUS_IF_FAILED(IallgatherBufferData(i,
                                                       send_buffers[i],
                                                       local_buffer_sizes[i],
                                                       receive_buf->GetByteBuffer(),
                                                       receive_counts,
                                                       disp_per_buffer));
    displacements->push_back(std::move(disp_per_buffer));
    received_buffers->push_back(std::move(receive_buf));
  }

  return WaitAll(num_buffers);
}

Status DoTableAllgather(TableAllgatherImpl &impl,
                        const std::shared_ptr<Table> &table,
                        std::vector<std::shared_ptr<Table>> *out) {
  std::shared_ptr<TableSerializer> serializer;
  RETURN_CYLON_STATUS_IF_FAILED(CylonTableSerializer::Make(table, &serializer));
  const auto &ctx = table->GetContext();
  auto *pool = ToArrowPool(ctx);

  const auto &allocator = std::make_shared<ArrowAllocator>(pool);
  std::vector<std::shared_ptr<Buffer>> receive_buffers;

  std::vector<int32_t> buffer_sizes_per_table;
  //  |b_0, ..., b_n-1|...|b_0, ..., b_n-1|
  //   <--- tbl_0 --->     <--- tbl_m --->

  std::vector<std::vector<int32_t>> all_disps;
  //  |t_0, ..., t_m-1|...|t_0, ..., t_m-1|
  //   <--- buf_0 --->     <--- buf_n --->

  RETURN_CYLON_STATUS_IF_FAILED(impl.Execute(serializer,
                                             allocator,
                                             ctx->GetWorldSize(),
                                             &buffer_sizes_per_table,
                                             &receive_buffers,
                                             &all_disps));


  // need to reshape all_disps for per-table basis
  auto buffer_offsets_per_table = ReshapeDispToPerTable(all_disps);

  const int num_tables = (int) all_disps[0].size();
  return DeserializeTables(ctx, table->get_table()->schema(), num_tables, receive_buffers,
                           buffer_sizes_per_table, buffer_offsets_per_table, out);
}

Status TableGatherImpl::Execute(const std::shared_ptr<cylon::TableSerializer> &serializer,
                                const std::shared_ptr<cylon::Allocator> &allocator,
                                int rank,
                                int world_size,
                                int gather_root,
                                bool gather_from_root,
                                std::vector<int32_t> *all_buffer_sizes,
                                std::vector<std::shared_ptr<cylon::Buffer>> *received_buffers,
                                std::vector<std::vector<int32_t>> *displacements) {
  int num_buffers = serializer->getNumberOfBuffers();

  // init comp
  Init(num_buffers);

  bool is_root = gather_root == rank;
  // first gather table buffer sizes
  std::vector<int32_t> local_buffer_sizes;
  if (is_root && !gather_from_root) {
    local_buffer_sizes = serializer->getEmptyTableBufferSizes();
  } else {
    local_buffer_sizes = serializer->getBufferSizes();
  }

  // gather size buffers
  if (is_root) {
    all_buffer_sizes->resize(world_size * num_buffers);
  }

  RETURN_CYLON_STATUS_IF_FAILED(GatherBufferSizes(local_buffer_sizes.data(), num_buffers,
                                                  all_buffer_sizes->data(), gather_root));

  std::vector<int32_t> total_buffer_sizes;
  if (is_root) {
    total_buffer_sizes = totalBufferSizes(*all_buffer_sizes, num_buffers, world_size);
  }

  const std::vector<const uint8_t *> &send_buffers = serializer->getDataBuffers();

  displacements->reserve(num_buffers);
  received_buffers->reserve(num_buffers);
  for (int32_t i = 0; i < num_buffers; ++i) {
    if (is_root) {
      std::shared_ptr<cylon::Buffer> receive_buf;
      RETURN_CYLON_STATUS_IF_FAILED(allocator->Allocate(total_buffer_sizes[i], &receive_buf));
      const auto &receive_counts = receiveCounts(*all_buffer_sizes, i,
                                                 num_buffers, world_size);
      auto disp_per_buffer = displacementsPerBuffer(*all_buffer_sizes, i,
                                                    num_buffers, world_size);

      RETURN_CYLON_STATUS_IF_FAILED(IgatherBufferData(i,
                                                      send_buffers[i],
                                                      local_buffer_sizes[i],
                                                      receive_buf->GetByteBuffer(),
                                                      receive_counts,
                                                      disp_per_buffer,
                                                      gather_root));
      displacements->push_back(std::move(disp_per_buffer));
      received_buffers->push_back(std::move(receive_buf));
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

  return WaitAll(num_buffers);
}

Status DoTableGather(TableGatherImpl &impl,
                     const std::shared_ptr<Table> &table,
                     int gather_root,
                     bool gather_from_root,
                     std::vector<std::shared_ptr<Table>> *out) {
  std::shared_ptr<TableSerializer> serializer;
  RETURN_CYLON_STATUS_IF_FAILED(CylonTableSerializer::Make(table, &serializer));
  const auto &ctx = table->GetContext();
  auto *pool = ToArrowPool(ctx);

  const auto &allocator = std::make_shared<ArrowAllocator>(pool);
  std::vector<std::shared_ptr<Buffer>> receive_buffers;

  std::vector<int32_t> buffer_sizes_per_table;
  //  |b_0, ..., b_n-1|...|b_0, ..., b_n-1|
  //   <--- tbl_0 --->     <--- tbl_m --->

  std::vector<std::vector<int32_t>> all_disps;
  //  |t_0, ..., t_m-1|...|t_0, ..., t_m-1|
  //   <--- buf_0 --->     <--- buf_n --->

  RETURN_CYLON_STATUS_IF_FAILED(impl.Execute(serializer,
                                             allocator,
                                             ctx->GetRank(),
                                             ctx->GetWorldSize(),
                                             gather_root,
                                             gather_from_root,
                                             &buffer_sizes_per_table,
                                             &receive_buffers,
                                             &all_disps));

  // need to reshape all_disps for per-table basis
  if (gather_root == ctx->GetRank()) {
    auto buffer_offsets_per_table = ReshapeDispToPerTable(all_disps);
    const int num_tables = (int) all_disps[0].size();
    return DeserializeTables(ctx, table->get_table()->schema(), num_tables, receive_buffers,
                             buffer_sizes_per_table, buffer_offsets_per_table, out);
  }
  return Status::OK();
}

}// net
}// cylon


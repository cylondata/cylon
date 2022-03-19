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

#include <arrow/ipc/api.h>
#include <arrow/io/memory.h>

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

Status TableBcastImpl::Execute(const std::shared_ptr<TableSerializer> &serializer,
                               const std::shared_ptr<Allocator> &allocator,
                               int32_t rank,
                               int32_t bcast_root,
                               std::vector<std::shared_ptr<Buffer>> *received_buffers,
                               std::vector<int32_t> *data_types) {
  bool is_root = rank == bcast_root;
// first broadcast the number of buffers
  int32_t num_buffers = 0;
  if (is_root) {
    num_buffers = serializer->getNumberOfBuffers();
  }

//  int status = MPI_Bcast(&num_buffers, 1, MPI_INT32_T, bcast_root, comm);
//  if (status != MPI_SUCCESS) {
//    return cylon::Status(cylon::Code::ExecutionError, "MPI_Bcast failed for size broadcast!");
//  }
  RETURN_CYLON_STATUS_IF_FAILED(BcastBufferSizes(&num_buffers, 1, bcast_root));

  // broadcast buffer sizes
  std::vector<int32_t>
      buffer_sizes = is_root ? serializer->getBufferSizes() : std::vector<int32_t>(num_buffers, 0);

  // init async ops
  Init(num_buffers);

//  status = MPI_Bcast(buffer_sizes.data(), num_buffers, MPI_INT32_T, bcast_root, comm);
//  if (status != MPI_SUCCESS) {
//    return cylon::Status(cylon::Code::ExecutionError, "MPI_Bcast failed for size array broadcast!");
//  }
  RETURN_CYLON_STATUS_IF_FAILED(BcastBufferSizes(buffer_sizes.data(), num_buffers, bcast_root));


  // broadcast data types
  *data_types = is_root ? serializer->getDataTypes() : std::vector<int32_t>(num_buffers / 3, 0);
//  if (is_root) {
//    data_types = serializer->getDataTypes();
//  } else {
//    data_types.resize(num_buffers / 3, 0);
//  }

//  status = MPI_Bcast(data_types.data(), data_types.size(), MPI_INT32_T, bcast_root, comm);
//  if (status != MPI_SUCCESS) {
//    return cylon::Status(cylon::Code::ExecutionError,
//                         "MPI_Bcast failed for data type array broadcast!");
//  }
  RETURN_CYLON_STATUS_IF_FAILED(BcastBufferSizes(data_types->data(),
                                                 data_types->size(),
                                                 bcast_root));


  // if all buffer sizes are zero, there are zero rows in the table
  // no need to broadcast any buffers
  if (std::all_of(buffer_sizes.begin(), buffer_sizes.end(), [](int32_t i) { return i == 0; })) {
    return cylon::Status::OK();
  }

//  std::vector<MPI_Request> requests(num_buffers);
//  std::vector<MPI_Status> statuses(num_buffers);
  std::vector<const uint8_t *> send_buffers{};
  if (is_root) {
    send_buffers = serializer->getDataBuffers();
  } else {
    received_buffers->reserve(num_buffers);
  }

  for (int32_t i = 0; i < num_buffers; ++i) {
    if (is_root) {
//      status = MPI_Ibcast(const_cast<uint8_t *>(send_buffers[i]),
//                          buffer_sizes[i],
//                          MPI_UINT8_T,
//                          bcast_root,
//                          comm,
//                          &requests[i]);
//      if (status != MPI_SUCCESS) {
//        return cylon::Status(cylon::Code::ExecutionError, "MPI_Ibast failed!");
//      }
      RETURN_CYLON_STATUS_IF_FAILED(IbcastBufferData(i,
                                                     const_cast<uint8_t *>(send_buffers[i]),
                                                     buffer_sizes[i],
                                                     bcast_root));
    } else {
      std::shared_ptr<cylon::Buffer> receive_buf;
      RETURN_CYLON_STATUS_IF_FAILED(allocator->Allocate(buffer_sizes[i], &receive_buf));

//      status = MPI_Ibcast(receive_buf->GetByteBuffer(),
//                          buffer_sizes[i],
//                          MPI_UINT8_T,
//                          bcast_root,
//                          comm,
//                          &requests[i]);
//      if (status != MPI_SUCCESS) {
//        return cylon::Status(cylon::Code::ExecutionError, "MPI_Ibast failed!");
//      }
      RETURN_CYLON_STATUS_IF_FAILED(IbcastBufferData(i,
                                                     receive_buf->GetByteBuffer(),
                                                     buffer_sizes[i],
                                                     bcast_root));
      received_buffers->push_back(std::move(receive_buf));
    }
  }

//  status = MPI_Waitall(num_buffers, requests.data(), statuses.data());
//  if (status != MPI_SUCCESS) {
//    return cylon::Status(cylon::Code::ExecutionError, "MPI_Ibast failed when waiting!");
//  }
//
//  return cylon::Status::OK();

  return WaitAll(num_buffers);
}

Status BcastArrowSchema(TableBcastImpl &impl, std::shared_ptr<arrow::Schema> *schema,
                        int32_t bcast_root, bool is_root, arrow::MemoryPool *pool) {
  std::shared_ptr<arrow::Buffer> buf;
  if (is_root) {
    CYLON_ASSIGN_OR_RAISE(buf, arrow::ipc::SerializeSchema(**schema, pool))
  }

  // bcast buffer size
  int32_t buf_size = is_root ? (int32_t) buf->size() : 0;
  RETURN_CYLON_STATUS_IF_FAILED(impl.BcastBufferSizes(&buf_size, 1, bcast_root));

  if (!is_root) { // if not root, allocate a buffer for incoming data
    CYLON_ASSIGN_OR_RAISE(buf, arrow::AllocateBuffer(buf_size));
  }

  // now bcast buffer data
  RETURN_CYLON_STATUS_IF_FAILED(impl.BcastBufferData(buf->mutable_data(), buf_size, bcast_root));

  if (!is_root) {
    assert(buf->data());
    arrow::io::BufferReader buf_reader(std::move(buf));
    CYLON_ASSIGN_OR_RAISE(*schema, arrow::ipc::ReadSchema(&buf_reader, nullptr))
  }

  return Status::OK();
}

Status DoTableBcast(TableBcastImpl &impl, std::shared_ptr<Table> *table, int bcast_root,
                    const std::shared_ptr<CylonContext> &ctx) {
  std::shared_ptr<arrow::Schema> schema;
  bool is_root = bcast_root == ctx->GetRank();
  auto *pool = ToArrowPool(ctx);

  if (is_root) {
    auto atable = (*table)->get_table();
    schema = atable->schema();
  }

  // first, broadcast schema
  RETURN_CYLON_STATUS_IF_FAILED(BcastArrowSchema(impl, &schema, bcast_root, is_root, pool));

  std::shared_ptr<TableSerializer> serializer;
  if (is_root) {
    RETURN_CYLON_STATUS_IF_FAILED(CylonTableSerializer::Make(*table, &serializer));
  }
  const auto &allocator = std::make_shared<ArrowAllocator>(pool);
  std::vector<std::shared_ptr<Buffer>> receive_buffers;
  std::vector<int32_t> data_types;

  RETURN_CYLON_STATUS_IF_FAILED(impl.Execute(serializer,
                                             allocator,
                                             ctx->GetRank(),
                                             bcast_root,
                                             &receive_buffers,
                                             &data_types));

  if (!is_root) {
    if (receive_buffers.empty()) {
      std::shared_ptr<arrow::Table> atable;
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(cylon::util::MakeEmptyArrowTable(schema, &atable, pool));
      return Table::FromArrowTable(ctx, std::move(atable), *table);
    } else {
      assert((int) receive_buffers.size() == 3 * schema->num_fields());
      RETURN_CYLON_STATUS_IF_FAILED(DeserializeTable(ctx, schema, receive_buffers, table));
    }
  }
  return Status::OK();
}

}// net
}// cylon


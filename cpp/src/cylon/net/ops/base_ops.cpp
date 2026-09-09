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

#include <cstring>

#include <arrow/api.h>
#include <arrow/ipc/api.h>
#include <arrow/io/memory.h>

#include "base_ops.hpp"
#include "cylon/scalar.hpp"
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
  std::vector<std::vector<int>> all_recv_counts(num_buffers);
  for (int32_t i = 0; i < num_buffers; ++i) {
    std::shared_ptr<cylon::Buffer> receive_buf;
    RETURN_CYLON_STATUS_IF_FAILED(allocator->Allocate(total_buffer_sizes[i], &receive_buf));
    all_recv_counts[i] = receiveCounts(*all_buffer_sizes, i, num_buffers,
                                               world_size);
    auto disp_per_buffer = displacementsPerBuffer(*all_buffer_sizes, i, num_buffers,
                                                  world_size);

    RETURN_CYLON_STATUS_IF_FAILED(IallgatherBufferData(i,
                                                       send_buffers[i],
                                                       local_buffer_sizes[i],
                                                       receive_buf->GetByteBuffer(),
                                                       all_recv_counts[i],
                                                       disp_per_buffer));
    displacements->push_back(std::move(disp_per_buffer));
    received_buffers->push_back(std::move(receive_buf));
  }

  return WaitAll(num_buffers);
}

Status TableAllgatherImpl::Execute(const std::shared_ptr<Table> &table,
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

  RETURN_CYLON_STATUS_IF_FAILED(Execute(serializer,
                                        allocator,
                                        ctx->GetWorldSize(),
                                        &buffer_sizes_per_table,
                                        &receive_buffers,
                                        &all_disps));


  // Row counts travel with the buffers — see the note in TableGatherImpl::Execute.
  int32_t local_rows = static_cast<int32_t>(table->Rows());
  std::vector<int32_t> num_rows_per_table(ctx->GetWorldSize(), 0);
  RETURN_CYLON_STATUS_IF_FAILED(AllgatherBufferSizes(&local_rows, 1,
                                                     num_rows_per_table.data()));

  // need to reshape all_disps for per-table basis
  auto buffer_offsets_per_table = ReshapeDispToPerTable(all_disps);

  const int num_tables = (int) all_disps[0].size();
  return DeserializeTables(ctx, table->get_table()->schema(), num_tables, receive_buffers,
                           buffer_sizes_per_table, buffer_offsets_per_table, out,
                           num_rows_per_table);
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
  std::vector<std::vector<int>> all_recv_counts(num_buffers);
  for (int32_t i = 0; i < num_buffers; ++i) {
    if (is_root) {
      std::shared_ptr<cylon::Buffer> receive_buf;
      RETURN_CYLON_STATUS_IF_FAILED(allocator->Allocate(total_buffer_sizes[i], &receive_buf));
      all_recv_counts[i] = receiveCounts(*all_buffer_sizes, i,
                                                 num_buffers, world_size);
      auto disp_per_buffer = displacementsPerBuffer(*all_buffer_sizes, i,
                                                    num_buffers, world_size);

      RETURN_CYLON_STATUS_IF_FAILED(IgatherBufferData(i,
                                                      send_buffers[i],
                                                      local_buffer_sizes[i],
                                                      receive_buf->GetByteBuffer(),
                                                      all_recv_counts[i],
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

Status TableGatherImpl::Execute(const std::shared_ptr<Table> &table,
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

  RETURN_CYLON_STATUS_IF_FAILED(Execute(serializer, allocator, ctx->GetRank(), ctx->GetWorldSize(),
                                        gather_root, gather_from_root,
                                        &buffer_sizes_per_table, &receive_buffers, &all_disps));

  // Row counts travel with the buffers: a bit-packed boolean column has no
  // recoverable length, so a table of only boolean columns would otherwise
  // arrive empty.
  const bool is_root = gather_root == ctx->GetRank();
  int32_t local_rows =
      (is_root && !gather_from_root) ? 0 : static_cast<int32_t>(table->Rows());
  std::vector<int32_t> num_rows_per_table;
  if (is_root) {
    num_rows_per_table.resize(ctx->GetWorldSize(), 0);
  }
  RETURN_CYLON_STATUS_IF_FAILED(GatherBufferSizes(&local_rows, 1, num_rows_per_table.data(),
                                                  gather_root));

  // need to reshape all_disps for per-table basis
  if (is_root) {
    auto buffer_offsets_per_table = ReshapeDispToPerTable(all_disps);
    const int num_tables = (int) all_disps[0].size();
    return DeserializeTables(ctx, table->get_table()->schema(), num_tables, receive_buffers,
                             buffer_sizes_per_table, buffer_offsets_per_table, out,
                             num_rows_per_table);
  }
  return Status::OK();
}

namespace {

// Serialize a table to a self-describing Arrow IPC stream blob (schema + record
// batches) so a scatter receiver can deserialize with no pre-shared schema.
// Modeled on checkpoint/checkpoint_serializer.cpp.
Status SerializeTableToIpc(const std::shared_ptr<Table> &table,
                           std::shared_ptr<arrow::Buffer> *out) {
  const auto &arrow_table = table->get_table();
  if (!arrow_table) {
    return {Code::Invalid, "Scatter table has no underlying Arrow table"};
  }
  CYLON_ASSIGN_OR_RAISE(auto stream, arrow::io::BufferOutputStream::Create())
  CYLON_ASSIGN_OR_RAISE(auto writer,
                        arrow::ipc::MakeStreamWriter(stream, arrow_table->schema()))
  arrow::TableBatchReader batch_reader(*arrow_table);
  std::shared_ptr<arrow::RecordBatch> batch;
  while (true) {
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(batch_reader.ReadNext(&batch));
    if (!batch) break;
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(writer->WriteRecordBatch(*batch));
  }
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(writer->Close());
  CYLON_ASSIGN_OR_RAISE(*out, stream->Finish())
  return Status::OK();
}

// Inverse of SerializeTableToIpc: reconstruct a cylon::Table from an IPC blob.
Status DeserializeIpcToTable(const uint8_t *data, int32_t size,
                             const std::shared_ptr<CylonContext> &ctx,
                             std::shared_ptr<Table> *table) {
  if (data == nullptr || size == 0) {
    return {Code::Invalid, "Empty data for scatter deserialization"};
  }
  auto buffer = arrow::Buffer::Wrap(data, size);
  auto reader_input = std::make_shared<arrow::io::BufferReader>(std::move(buffer));
  CYLON_ASSIGN_OR_RAISE(auto reader,
                        arrow::ipc::RecordBatchStreamReader::Open(std::move(reader_input)))
  CYLON_ASSIGN_OR_RAISE(auto arrow_table, reader->ToTable())
  return Table::FromArrowTable(ctx, std::move(arrow_table), *table);
}

}  // namespace

Status TableScatterImpl::Execute(const std::vector<std::shared_ptr<Table>> &tables,
                                 int32_t scatter_root,
                                 const std::shared_ptr<CylonContext> &ctx,
                                 std::shared_ptr<Table> *out) {
  const int32_t world_size = ctx->GetWorldSize();
  const int32_t rank = ctx->GetRank();
  const bool is_root = rank == scatter_root;
  auto *a_pool = ToArrowPool(ctx);

  if (world_size == 1) {
    if (tables.empty()) {
      return {Code::Invalid, "Scatter root must provide world_size tables"};
    }
    *out = tables[0];
    return Status::OK();
  }

  Init(1);

  // Root serializes each shard to a self-describing Arrow IPC blob and records
  // its byte length.
  std::vector<std::shared_ptr<arrow::Buffer>> blobs;
  std::vector<int32_t> counts(world_size, 0);
  if (is_root) {
    if ((int32_t) tables.size() != world_size) {
      return {Code::Invalid, "Scatter root must provide exactly world_size tables"};
    }
    blobs.resize(world_size);
    for (int32_t r = 0; r < world_size; r++) {
      RETURN_CYLON_STATUS_IF_FAILED(SerializeTableToIpc(tables[r], &blobs[r]));
      counts[r] = (int32_t) blobs[r]->size();
    }
  }

  // Broadcast the per-rank byte counts so every rank sizes its recv buffer AND
  // agrees on the even-vs-uneven (scatter vs scatterv) choice identically — the
  // collective-type decision must be the same on all ranks.
  RETURN_CYLON_STATUS_IF_FAILED(BcastBufferSizes(counts.data(), world_size, scatter_root));

  // Prefix-sum displacements — identical on every rank now that counts is.
  std::vector<int32_t> displs(world_size, 0);
  for (int32_t r = 1; r < world_size; r++) {
    displs[r] = displs[r - 1] + counts[r - 1];
  }

  // Root packs all blobs contiguously into a single send buffer.
  std::shared_ptr<arrow::Buffer> send_buf;
  if (is_root) {
    int32_t total = displs[world_size - 1] + counts[world_size - 1];
    CYLON_ASSIGN_OR_RAISE(send_buf, arrow::AllocateBuffer(total, a_pool))
    for (int32_t r = 0; r < world_size; r++) {
      if (counts[r] > 0) {
        std::memcpy(send_buf->mutable_data() + displs[r], blobs[r]->data(), counts[r]);
      }
    }
  }

  const int32_t my_size = counts[rank];
  CYLON_ASSIGN_OR_RAISE(auto recv_buf, arrow::AllocateBuffer(my_size, a_pool))

  RETURN_CYLON_STATUS_IF_FAILED(IscatterBufferData(is_root ? send_buf->data() : nullptr,
                                                   counts, displs,
                                                   recv_buf->mutable_data(), my_size,
                                                   scatter_root));

  RETURN_CYLON_STATUS_IF_FAILED(WaitAll(1));

  return DeserializeIpcToTable(recv_buf->data(), my_size, ctx, out);
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
  RETURN_CYLON_STATUS_IF_FAILED(BcastBufferSizes(&num_buffers, 1, bcast_root));

  // init async ops
  Init(num_buffers);

  // broadcast buffer sizes
  std::vector<int32_t> buffer_sizes = is_root ? serializer->getBufferSizes()
                                              : std::vector<int32_t>(num_buffers, 0);
  RETURN_CYLON_STATUS_IF_FAILED(BcastBufferSizes(buffer_sizes.data(), num_buffers, bcast_root));

  // broadcast data types. The count travels on the wire rather than being derived
  // from num_buffers: there is one data type per column, but a column with a nested
  // type occupies more than three buffer slots, so num_buffers / 3 would leave root
  // and non-root disagreeing on how many elements this broadcast moves.
  if (is_root) {
    *data_types = serializer->getDataTypes();
  }
  int32_t num_data_types = is_root ? static_cast<int32_t>(data_types->size()) : 0;
  RETURN_CYLON_STATUS_IF_FAILED(BcastBufferSizes(&num_data_types, 1, bcast_root));
  if (!is_root) {
    data_types->assign(num_data_types, 0);
  }
  RETURN_CYLON_STATUS_IF_FAILED(BcastBufferSizes(data_types->data(), num_data_types, bcast_root));

  // if all buffer sizes are zero, there are zero rows in the table
  // no need to broadcast any buffers
  if (std::all_of(buffer_sizes.begin(), buffer_sizes.end(), [](int32_t i) { return i == 0; })) {
    return cylon::Status::OK();
  }

  std::vector<const uint8_t *> send_buffers{};
  if (is_root) {
    send_buffers = serializer->getDataBuffers();
  } else {
    received_buffers->reserve(num_buffers);
  }

  for (int32_t i = 0; i < num_buffers; ++i) {
    if (is_root) {
      RETURN_CYLON_STATUS_IF_FAILED(IbcastBufferData(i, const_cast<uint8_t *>(send_buffers[i]),
                                                     buffer_sizes[i], bcast_root));
    } else {
      std::shared_ptr<cylon::Buffer> receive_buf;
      RETURN_CYLON_STATUS_IF_FAILED(allocator->Allocate(buffer_sizes[i], &receive_buf));
      RETURN_CYLON_STATUS_IF_FAILED(IbcastBufferData(i, receive_buf->GetByteBuffer(),
                                                     buffer_sizes[i], bcast_root));
      received_buffers->push_back(std::move(receive_buf));
    }
  }
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

Status TableBcastImpl::Execute(std::shared_ptr<Table> *table, int bcast_root,
                               const std::shared_ptr<CylonContext> &ctx) {
  std::shared_ptr<arrow::Schema> schema;
  bool is_root = bcast_root == ctx->GetRank();
  auto *pool = ToArrowPool(ctx);

  if (is_root) {
    auto atable = (*table)->get_table();
    schema = atable->schema();
  }

  // first, broadcast schema
  RETURN_CYLON_STATUS_IF_FAILED(BcastArrowSchema(*this, &schema, bcast_root, is_root, pool));

  int32_t num_rows = is_root ? static_cast<int32_t>((*table)->Rows()) : 0;
  RETURN_CYLON_STATUS_IF_FAILED(BcastBufferSizes(&num_rows, 1, bcast_root));

  std::shared_ptr<TableSerializer> serializer;
  if (is_root) {
    RETURN_CYLON_STATUS_IF_FAILED(CylonTableSerializer::Make(*table, &serializer));
  }
  const auto &allocator = std::make_shared<ArrowAllocator>(pool);
  std::vector<std::shared_ptr<Buffer>> receive_buffers;
  std::vector<int32_t> data_types;

  RETURN_CYLON_STATUS_IF_FAILED(Execute(serializer,
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
      assert((int) receive_buffers.size() == BufferSlots(schema));
      RETURN_CYLON_STATUS_IF_FAILED(DeserializeTable(ctx, schema, receive_buffers, table,
                                                     num_rows));
    }
  }
  return Status::OK();
}

Status AllReduceImpl::Execute(const std::shared_ptr<Column> &values,
                              net::ReduceOp reduce_op,
                              std::shared_ptr<Column> *output,
                              MemoryPool *pool) const {
  auto a_pool = ToArrowPool(pool);
  const auto &arr = values->data();

  auto arrow_t = arr->data()->type;
  int byte_width = arrow::bit_width(arrow_t->id()) / 8;

  if (byte_width == 0) {
    return {Code::Invalid, "Allreduce does not support " + arrow_t->ToString()};
  }

  // all ranks should have 0 null count, and equal size.
  // equal size can be checked using this trick https://stackoverflow.com/q/71161571/4116268
  std::array<int64_t, 3> metadata{arr->null_count(), arr->length(), -arr->length()};
  std::array<int64_t, 3> metadata_res{0, 0, 0};

  RETURN_CYLON_STATUS_IF_FAILED(AllReduceBuffer(metadata.data(), metadata_res.data(), 3,
                                                Int64(), MAX));

  if (metadata_res[0] > 0) {
    return {Code::Invalid, "Allreduce does not support null values"};
  }
  if (metadata_res[1] != -metadata[2]) {
    return {Code::Invalid, "Allreduce values should be the same length in all ranks"};
  }

  int count = static_cast<int>(arr->length());
  int64_t nbytes = (int64_t) byte_width * count;

  // Reduce into a SCRATCH copy of the input, not the input itself: the FMI direct
  // (PeerToPeer) reduce combines in place into the send buffer, which would otherwise
  // corrupt the caller's nominally-immutable input Arrow column. redis/UCC don't mutate
  // the send buffer, so this copy is a harmless no-op-correctness cost there.
  CYLON_ASSIGN_OR_RAISE(auto send_buf, arrow::AllocateBuffer(nbytes, a_pool))
  std::memcpy(send_buf->mutable_data(), arr->data()->GetValues<uint8_t>(1), nbytes);
  CYLON_ASSIGN_OR_RAISE(auto buf, arrow::AllocateBuffer(nbytes, a_pool))

  RETURN_CYLON_STATUS_IF_FAILED(
      AllReduceBuffer(send_buf->data(), buf->mutable_data(), count,
                      values->type(), reduce_op));

  *output = Column::Make(arrow::MakeArray(arrow::ArrayData::Make(std::move(arrow_t),
                                                                 count,
                                                                 {nullptr, std::move(buf)},
                                                                 0, 0)));
  return Status::OK();
}

Status AllReduceImpl::Execute(const std::shared_ptr<Scalar> &value,
                              net::ReduceOp reduce_op,
                              std::shared_ptr<Scalar> *output,
                              MemoryPool *pool) const {
  CYLON_ASSIGN_OR_RAISE(auto arr,
                        arrow::MakeArrayFromScalar(*value->data(), 1, ToArrowPool(pool)))
  const auto &col = Column::Make(std::move(arr));
  std::shared_ptr<Column> out_arr;
  RETURN_CYLON_STATUS_IF_FAILED(Execute(col, reduce_op, &out_arr, pool));
  CYLON_ASSIGN_OR_RAISE(auto out_scal, out_arr->data()->GetScalar(0))
  *output = Scalar::Make(std::move(out_scal));
  return Status::OK();
}

Status ReduceImpl::Execute(const std::shared_ptr<Column> &values,
                           net::ReduceOp reduce_op,
                           int reduce_root,
                           std::shared_ptr<Column> *output,
                           MemoryPool *pool) const {
  auto a_pool = ToArrowPool(pool);
  const auto &arr = values->data();

  auto arrow_t = arr->data()->type;
  int byte_width = arrow::bit_width(arrow_t->id()) / 8;

  if (byte_width == 0) {
    return {Code::Invalid, "Reduce does not support " + arrow_t->ToString()};
  }

  // Unlike AllReduce we skip the cross-rank metadata validation (there is no
  // allreduce available inside a reduce). All ranks must supply the same length
  // per the element-wise reduce contract; the reduced buffer is only meaningful
  // at `reduce_root` — non-root output is undefined by contract.
  int count = static_cast<int>(arr->length());
  int64_t nbytes = (int64_t) byte_width * count;

  // Reduce into a SCRATCH copy of the input, not the input itself: the FMI direct
  // (PeerToPeer) reduce combines in place into the send buffer, which would otherwise
  // corrupt the caller's nominally-immutable input Arrow column. redis/UCC don't mutate
  // the send buffer, so this copy is a harmless correctness cost there.
  CYLON_ASSIGN_OR_RAISE(auto send_buf, arrow::AllocateBuffer(nbytes, a_pool))
  std::memcpy(send_buf->mutable_data(), arr->data()->GetValues<uint8_t>(1), nbytes);
  CYLON_ASSIGN_OR_RAISE(auto buf, arrow::AllocateBuffer(nbytes, a_pool))

  RETURN_CYLON_STATUS_IF_FAILED(
      ReduceBuffer(send_buf->data(), buf->mutable_data(), count,
                   values->type(), reduce_op, reduce_root));

  *output = Column::Make(arrow::MakeArray(arrow::ArrayData::Make(std::move(arrow_t),
                                                                 count,
                                                                 {nullptr, std::move(buf)},
                                                                 0, 0)));
  return Status::OK();
}

Status ReduceImpl::Execute(const std::shared_ptr<Scalar> &value,
                           net::ReduceOp reduce_op,
                           int reduce_root,
                           std::shared_ptr<Scalar> *output,
                           MemoryPool *pool) const {
  CYLON_ASSIGN_OR_RAISE(auto arr,
                        arrow::MakeArrayFromScalar(*value->data(), 1, ToArrowPool(pool)))
  const auto &col = Column::Make(std::move(arr));
  std::shared_ptr<Column> out_arr;
  RETURN_CYLON_STATUS_IF_FAILED(Execute(col, reduce_op, reduce_root, &out_arr, pool));
  CYLON_ASSIGN_OR_RAISE(auto out_scal, out_arr->data()->GetScalar(0))
  *output = Scalar::Make(std::move(out_scal));
  return Status::OK();
}

void prefix_sum(const std::vector<int32_t> &buff_sizes, std::vector<int32_t> *out) {
  std::partial_sum(buff_sizes.begin(), buff_sizes.end() - 1, out->begin() + 1);
}

Status AllGatherImpl::Execute(const std::shared_ptr<Column> &values,
                              int32_t world_size,
                              std::vector<std::shared_ptr<Column>> *output,
                              MemoryPool *pool) {
  if (world_size == 1) {
    *output = {values};
    return Status::OK();
  }

  const auto &type = values->data()->type();
  std::shared_ptr<ColumnSerializer> serializer;
  RETURN_CYLON_STATUS_IF_FAILED(CylonColumnSerializer::Make(values, &serializer, pool));

  const auto &buf_sizes = serializer->buffer_sizes();
  const auto &buffers = serializer->data_buffers();

  // |b_0, b_1, b_2|...|b_0, b_1, b_2|
  // <----col@0---->   <---col@n-1--->
  std::vector<int32_t> all_buf_sizes(world_size * 3);
  RETURN_CYLON_STATUS_IF_FAILED(AllgatherBufferSize(buf_sizes.data(), 3, all_buf_sizes.data()));

  std::array<int32_t, 3> total_buf_sizes{};
  for (int i = 0; i < world_size; i++) {
    total_buf_sizes[0] += all_buf_sizes[3 * i];
    total_buf_sizes[1] += all_buf_sizes[3 * i + 1];
    total_buf_sizes[2] += all_buf_sizes[3 * i + 2];
  }

  ArrowAllocator allocator(ToArrowPool(pool));

  std::array<std::vector<int32_t>, 3> displacements{};
  std::array<std::shared_ptr<Buffer>, 3> received_bufs{};
  std::vector<std::vector<int>> all_recv_counts(3);
  for (int i = 0; i < 3; i++) {
    RETURN_CYLON_STATUS_IF_FAILED(allocator.Allocate(total_buf_sizes[i], &received_bufs[i]));

    all_recv_counts[i] = receiveCounts(all_buf_sizes, i, 3, world_size);
    displacements[i].resize(world_size);
    prefix_sum(all_recv_counts[i], &displacements[i]);

    RETURN_CYLON_STATUS_IF_FAILED(IallgatherBufferData(
        i, buffers[i], buf_sizes[i], received_bufs[i]->GetByteBuffer(),
        all_recv_counts[i], displacements[i]));
  }
  WaitAll();

  output->resize(world_size);
  for (int i = 0; i < world_size; i++) {
    std::array<int32_t, 3> sizes{all_buf_sizes[3 * i], all_buf_sizes[3 * i + 1],
                                 all_buf_sizes[3 * i + 2]};
    std::array<int32_t, 3> offsets{displacements[0][i], displacements[1][i], displacements[2][i]};
    RETURN_CYLON_STATUS_IF_FAILED(DeserializeColumn(type,
                                                    received_bufs,
                                                    sizes,
                                                    offsets,
                                                    &(*output)[i]));
  }
  return Status::OK();
}

Status AllGatherImpl::Execute(const std::shared_ptr<Scalar> &value,
                              int32_t world_size,
                              std::shared_ptr<Column> *output,
                              MemoryPool *pool) {
  auto a_pool = ToArrowPool(pool);
  CYLON_ASSIGN_OR_RAISE(auto arr, arrow::MakeArrayFromScalar(*value->data(), 1, a_pool));
  std::vector<std::shared_ptr<Column>> columns;
  RETURN_CYLON_STATUS_IF_FAILED(Execute(Column::Make(std::move(arr)), world_size, &columns, pool));

  std::vector<std::shared_ptr<arrow::Array>> a_arrs;
  a_arrs.reserve(world_size);
  for (const auto &c: columns) {
    a_arrs.push_back(c->data());
  }
  CYLON_ASSIGN_OR_RAISE(auto a_res, arrow::Concatenate(a_arrs, a_pool))

  *output = Column::Make(std::move(a_res));
  return Status::OK();
}

}// net
}// cylon


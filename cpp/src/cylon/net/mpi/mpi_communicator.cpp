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

#include <memory>

#include <cylon/net/communicator.hpp>
#include <cylon/net/mpi/mpi_communicator.hpp>
#include <cylon/net/mpi/mpi_channel.hpp>
#include <cylon/util/macros.hpp>

#include "cylon/arrow/arrow_buffer.hpp"
#include "cylon/util/arrow_utils.hpp"
#include "cylon/serialize/table_serialize.hpp"
#include "cylon/net/mpi/mpi_operations.hpp"
#include "cylon/scalar.hpp"

#include <arrow/ipc/api.h>
#include <arrow/io/api.h>

namespace cylon {
namespace net {

// configs
CommType MPIConfig::Type() {
  return CommType::MPI;
}

std::shared_ptr<MPIConfig> MPIConfig::Make(MPI_Comm comm) {
  return std::make_shared<MPIConfig>(comm);
}

MPIConfig::MPIConfig(MPI_Comm comm) : comm_(comm) {}

MPI_Comm MPIConfig::GetMPIComm() const {
  return comm_;
}

MPIConfig::~MPIConfig() = default;

std::unique_ptr<Channel> MPICommunicator::CreateChannel() const {
  return std::make_unique<MPIChannel>(mpi_comm_);
}

int MPICommunicator::GetRank() const {
  return this->rank;
}
int MPICommunicator::GetWorldSize() const {
  return this->world_size;
}
Status MPICommunicator::Init(const std::shared_ptr<CommConfig> &config) {
  // check if MPI is initialized
  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Initialized(&mpi_initialized_externally));
  mpi_comm_ = std::static_pointer_cast<MPIConfig>(config)->GetMPIComm();

  if (mpi_comm_ && !mpi_initialized_externally) {
    return {Code::Invalid, "non-nullptr MPI_Comm passed without initializing MPI"};
  }

  if (!mpi_initialized_externally) { // if not initialized, init MPI
    RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Init(nullptr, nullptr));
  }

  if (!mpi_comm_) { // set comm_ to world
    mpi_comm_ = MPI_COMM_WORLD;
  }

  // setting errors to return
  MPI_Comm_set_errhandler(mpi_comm_, MPI_ERRORS_RETURN);

  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Comm_rank(mpi_comm_, &this->rank));
  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Comm_size(mpi_comm_, &this->world_size));

  if (rank < 0 || world_size < 0 || rank >= world_size) {
    return {Code::ExecutionError, "Malformed rank :" + std::to_string(rank)
        + " or world size:" + std::to_string(world_size)};
  }

  return Status::OK();
}

void MPICommunicator::Finalize() {
  // finalize only if we initialized MPI
  if (!mpi_initialized_externally) {
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
  }
}
void MPICommunicator::Barrier() {
  MPI_Barrier(mpi_comm_);
}

CommType MPICommunicator::GetCommType() const {
  return MPI;
}

/*
    |t_0, ..., t_m-1|...|t_0, ..., t_m-1|
     <--- buf_0 --->     <--- buf_n --->
                  to
    |b_0, ..., b_n-1|...|b_0, ..., b_n-1|
     <--- tbl_0 --->     <--- tbl_m --->
 */
std::vector<int32_t> ReshapeDispToPerTable(const std::vector<std::vector<int32_t>> &all_disps) {
  const size_t num_buf = all_disps.size();
  const size_t num_tables = all_disps[0].size(); // == world_size

  std::vector<int32_t> res(num_buf * num_tables, 0);
  for (size_t tid = 0; tid < num_tables; tid++) {
    for (size_t bid = 0; bid < num_buf; bid++) {
      res[tid * num_buf + bid] = all_disps[bid][tid];
    }
  }
  return res;
}

Status MPICommunicator::AllGather(const std::shared_ptr<Table> &table,
                                  std::vector<std::shared_ptr<Table>> *out) const {
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

  RETURN_CYLON_STATUS_IF_FAILED(mpi::AllGather(serializer, allocator, buffer_sizes_per_table,
                                               receive_buffers, all_disps, ctx));

  // need to reshape all_disps for per-table basis
  auto buffer_offsets_per_table = ReshapeDispToPerTable(all_disps);

  const int num_tables = (int) all_disps[0].size();
  return DeserializeTables(ctx, table->get_table()->schema(), num_tables, receive_buffers,
                           buffer_sizes_per_table, buffer_offsets_per_table, out);
}

Status MPICommunicator::Gather(const std::shared_ptr<Table> &table,
                               int gather_root,
                               bool gather_from_root,
                               std::vector<std::shared_ptr<Table>> *out) const {
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

  RETURN_CYLON_STATUS_IF_FAILED(mpi::Gather(serializer, gather_root, gather_from_root,
                                            allocator, buffer_sizes_per_table, receive_buffers,
                                            all_disps, ctx));

  // need to reshape all_disps for per-table basis
  if (mpi::AmIRoot(gather_root, ctx)) {
    auto buffer_offsets_per_table = ReshapeDispToPerTable(all_disps);

    const int num_tables = (int) all_disps[0].size();
    return DeserializeTables(ctx, table->get_table()->schema(), num_tables, receive_buffers,
                             buffer_sizes_per_table, buffer_offsets_per_table, out);
  }
  return Status::OK();
}

Status BcastArrowSchema(const std::shared_ptr<CylonContext> &ctx,
                        std::shared_ptr<arrow::Schema> *schema,
                        int bcast_root) {
  std::shared_ptr<arrow::Buffer> buf;
  bool is_root = mpi::AmIRoot(bcast_root, ctx);

  if (is_root) {
    CYLON_ASSIGN_OR_RAISE(buf, arrow::ipc::SerializeSchema(**schema,
                                                           ToArrowPool(ctx)))
  }
  RETURN_CYLON_STATUS_IF_FAILED(mpi::BcastArrowBuffer(buf, bcast_root, ctx));

  if (!is_root) {
    assert(buf->data());
    arrow::io::BufferReader buf_reader(std::move(buf));
    CYLON_ASSIGN_OR_RAISE(*schema, arrow::ipc::ReadSchema(&buf_reader, nullptr))
  }

  return Status::OK();
}

Status MPICommunicator::Bcast(std::shared_ptr<Table> *table, int bcast_root) const {
  std::shared_ptr<arrow::Schema> schema;
  bool is_root = mpi::AmIRoot(bcast_root, *ctx_ptr);
  auto *pool = ToArrowPool(*ctx_ptr);

  if (is_root) {
    auto atable = (*table)->get_table();
    schema = atable->schema();
  }
  RETURN_CYLON_STATUS_IF_FAILED(BcastArrowSchema(*ctx_ptr, &schema, bcast_root));

  std::shared_ptr<TableSerializer> serializer;
  if (is_root) {
    RETURN_CYLON_STATUS_IF_FAILED(CylonTableSerializer::Make(*table, &serializer));
  }
  const auto &allocator = std::make_shared<ArrowAllocator>(pool);
  std::vector<std::shared_ptr<Buffer>> receive_buffers;
  std::vector<int32_t> data_types;
  RETURN_CYLON_STATUS_IF_FAILED(
      mpi::Bcast(serializer, bcast_root, allocator, receive_buffers, data_types, *ctx_ptr));

  if (!is_root) {
    if (receive_buffers.empty()) {
      std::shared_ptr<arrow::Table> atable;
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(cylon::util::MakeEmptyArrowTable(schema, &atable, pool));
      return Table::FromArrowTable(*ctx_ptr, std::move(atable), *table);
    } else {
      assert((int) receive_buffers.size() == 3 * schema->num_fields());
      RETURN_CYLON_STATUS_IF_FAILED(DeserializeTable(*ctx_ptr, schema, receive_buffers, table));
    }
  }
  return Status::OK();
}

MPI_Comm MPICommunicator::mpi_comm() const {
  return mpi_comm_;
}

Status MPICommunicator::AllReduce(const std::shared_ptr<Column> &values,
                                  net::ReduceOp reduce_op,
                                  std::shared_ptr<Column> *output) const {
  auto *pool = ToArrowPool(*ctx_ptr);
  const auto &arr = values->data();

  auto arrow_t = arr->data()->type;
  int byte_width = arrow::bit_width(arrow_t->id()) / 8;

  if (byte_width == 0) {
    return {Code::Invalid, "Allreduce does not support " + arrow_t->ToString()};
  }

  // all ranks should have 0 null count, and equal size.
  // equal size can be checked using this trick https://stackoverflow.com/q/71161571/4116268
  std::array<int64_t, 3> metadata{arr->null_count(), arr->length(), -arr->length()};
  RETURN_CYLON_STATUS_IF_MPI_FAILED(
      MPI_Allreduce(MPI_IN_PLACE, metadata.data(), 3, MPI_INT64_T, MPI_MAX, mpi_comm_));
  if (metadata[0] > 0) {
    return {Code::Invalid, "Allreduce does not support null values"};
  }
  if (metadata[1] != -metadata[2]) {
    return {Code::Invalid, "Allreduce values should be the same length in all ranks"};
  }

  int count = static_cast<int>(arr->length());
  CYLON_ASSIGN_OR_RAISE(auto buf, arrow::AllocateBuffer(byte_width * count, pool))
  RETURN_CYLON_STATUS_IF_FAILED(
      mpi::AllReduce(*ctx_ptr, arr->data()->GetValues<uint8_t>(1), buf->mutable_data(), count,
                     values->type(), reduce_op));

  *output = Column::Make(arrow::MakeArray(arrow::ArrayData::Make(std::move(arrow_t),
                                                                 count,
                                                                 {nullptr, std::move(buf)},
                                                                 0, 0)));
  return Status::OK();
}

Status MPICommunicator::AllReduce(const std::shared_ptr<Scalar> &value, net::ReduceOp reduce_op,
                                  std::shared_ptr<Scalar> *output) const {
  CYLON_ASSIGN_OR_RAISE(auto arr,
                        arrow::MakeArrayFromScalar(*value->data(), 1, ToArrowPool(*ctx_ptr)))
  const auto &col = Column::Make(std::move(arr));
  std::shared_ptr<Column> out_arr;
  RETURN_CYLON_STATUS_IF_FAILED(AllReduce(col, reduce_op, &out_arr));
  CYLON_ASSIGN_OR_RAISE(auto out_scal, out_arr->data()->GetScalar(0))
  *output = Scalar::Make(std::move(out_scal));
  return Status::OK();
}

MPICommunicator::MPICommunicator(const std::shared_ptr<CylonContext> *ctx_ptr)
    : Communicator(ctx_ptr) {}

}  // namespace net
}  // namespace cylon

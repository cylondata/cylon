#include <cylon/net/ucc/ucc_operations.hpp>
#include <cylon/util/macros.hpp>

namespace cylon {
namespace ucc {

Status UccTableAllgatherImpl::AllgatherBufferSizes(const int32_t *send_data,
                                                   int num_buffers,
                                                   int32_t *rcv_data) const {
  ucc_coll_args_t args;
  ucc_coll_req_h req;

  args.mask = 0;
  args.coll_type = UCC_COLL_TYPE_ALLGATHER;

  args.src.info.buffer = const_cast<int32_t *>(send_data);
  args.src.info.count = static_cast<uint64_t>(num_buffers);
  args.src.info.datatype = UCC_DT_INT32;
  args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
  args.dst.info.buffer = rcv_data;
  args.dst.info.count = static_cast<uint64_t>(num_buffers * world_size);
  args.dst.info.datatype = UCC_DT_INT32;
  args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_collective_init(&args, &req, ucc_team_));
  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_collective_post(req));

  ucc_status_t status;

  while (UCC_OK != (status = ucc_collective_test(req))) {
    RETURN_CYLON_STATUS_IF_UCC_FAILED(status);
    RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_context_progress(ucc_context_));
  }

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_collective_finalize(req));

  return Status::OK();
}

Status UccTableAllgatherImpl::IallgatherBufferData(
    int buf_idx, const uint8_t *send_data, int32_t send_count,
    uint8_t *recv_data, const std::vector<int32_t> &recv_count,
    const std::vector<int32_t> &displacements) {
  ucc_coll_args_t &args = args_[buf_idx];

  args.mask = 0;
  args.coll_type = UCC_COLL_TYPE_ALLGATHERV;

  args.src.info.buffer = const_cast<uint8_t *>(send_data);
  args.src.info.count = static_cast<uint64_t>(send_count);
  args.src.info.datatype = UCC_DT_UINT8;
  args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

  args.dst.info_v.buffer = recv_data;
  args.dst.info_v.counts = (ucc_count_t *) recv_count.data();
  args.dst.info_v.displacements = (ucc_aint_t *) displacements.data();
  args.dst.info_v.datatype = UCC_DT_UINT8;
  args.dst.info_v.mem_type = UCC_MEMORY_TYPE_HOST;

  RETURN_CYLON_STATUS_IF_UCC_FAILED(
      ucc_collective_init(&args, &requests_[buf_idx], ucc_team_));

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_collective_post(requests_[buf_idx]));
  return Status::OK();
}

ucc_status_t test(std::vector<ucc_coll_req_h>& reqs) {
  ucc_status_t status = UCC_OK;
  for (auto r : reqs) {
    status = ucc_collective_test(r);
    if (UCC_OK != status && UCC_OPERATION_INITIALIZED != status) {
      break;
    }
  }
  return status;
}

ucc_status_t WaitAllHelper(std::vector<ucc_coll_req_h>& reqs, ucc_context_h& ctx) {
  bool alldone = false;
  ucc_status_t status;
  while (!alldone) {
    alldone = true;
    for (auto &r : reqs) {
      if (UCC_OK != (status = test(reqs))) {
        if (status < 0) {
          return status;
        }
        alldone = false;
        ucc_context_progress(ctx);
      }
    }
  }
  return UCC_OK;
}

Status UccTableAllgatherImpl::WaitAll(int num_buffers) {
  CYLON_UNUSED(num_buffers);
  RETURN_CYLON_STATUS_IF_UCC_FAILED(WaitAllHelper(requests_, ucc_context_));
  return Status::OK();
}

UccTableAllgatherImpl::UccTableAllgatherImpl(ucc_team_h ucc_team,
                                             ucc_context_h ucc_context,
                                             int ws)
    : TableAllgatherImpl(),
      ucc_team_(ucc_team),
      ucc_context_(ucc_context),
      requests_({}),
      args_({}),
      world_size(ws){}

void UccTableAllgatherImpl::Init(int num_buffers) {
  requests_.resize(num_buffers);
  args_.resize(num_buffers);
}

UccTableAllgatherImpl::~UccTableAllgatherImpl() {
  for (auto req: requests_) {
    ucc_collective_finalize(req);
  }
}

ucc_datatype_t GetUccDataType(const std::shared_ptr<DataType> &data_type) {
  switch (data_type->getType()) {
    case Type::BOOL:

      break;
    case Type::UINT8:
      return UCC_DT_UINT8;
    case Type::INT8:
      return UCC_DT_INT8;
    case Type::UINT16:
      return UCC_DT_UINT16;
    case Type::INT16:
      return UCC_DT_INT16;
    case Type::UINT32:
      return UCC_DT_UINT32;
    case Type::INT32:
      return UCC_DT_INT32;
    case Type::UINT64:
      return UCC_DT_UINT64;
    case Type::INT64:
      return UCC_DT_INT64;
    case Type::FLOAT:
      return UCC_DT_FLOAT32;
    case Type::DOUBLE:
      return UCC_DT_FLOAT64;
    case Type::FIXED_SIZE_BINARY:
    case Type::STRING:
    case Type::BINARY:
    case Type::LARGE_STRING:
    case Type::LARGE_BINARY:
      return UCC_DT_UINT8;
      // todo: MPI does not support 16byte floats. We'll have to use a custom
      // datatype for this later.
    case Type::HALF_FLOAT:
      return UCC_DT_FLOAT16;
    case Type::DATE32:
      return UCC_DT_UINT32;
    case Type::DATE64:
      return UCC_DT_UINT64;
    case Type::TIMESTAMP:
      return UCC_DT_UINT64;
    case Type::TIME32:
      return UCC_DT_UINT32;
    case Type::TIME64:
      return UCC_DT_UINT64;
    case Type::DECIMAL:
    case Type::DURATION:
    case Type::INTERVAL:
    case Type::LIST:
    case Type::FIXED_SIZE_LIST:
    case Type::EXTENSION:
      break;
    case Type::MAX_ID:
      break;
  }
  return UCC_DT_PREDEFINED_LAST;
}

ucc_reduction_op_t GetUccOp(cylon::net::ReduceOp reduce_op) {
  switch (reduce_op) {
    case net::SUM:
      return UCC_OP_SUM;
    case net::MIN:
      return UCC_OP_MIN;
    case net::MAX:
      return UCC_OP_MAX;
    case net::PROD:
      return UCC_OP_PROD;
    case net::LAND:
      return UCC_OP_LAND;
    case net::LOR:
      return UCC_OP_LOR;
    case net::BAND:
      return UCC_OP_BAND;
    case net::BOR:
      return UCC_OP_BOR;
    default:
      return UCC_OP_LAST;
  }
}

UccAllReduceImpl::UccAllReduceImpl(ucc_team_h ucc_team,
                                   ucc_context_h ucc_context, int ws)
    : ucc_team_(ucc_team), ucc_context_(ucc_context), world_size(ws) {}

Status UccAllReduceImpl::AllReduceBuffer(const void *send_buf, void *rcv_buf,
                                  int count,
                                  const std::shared_ptr<DataType> &data_type,
                                  cylon::net::ReduceOp reduce_op) const {
  auto dt = GetUccDataType(data_type);
  auto op = GetUccOp(reduce_op);

  if (dt == UCC_DT_PREDEFINED_LAST || op == UCC_OP_LAST) {
    return {Code::NotImplemented, "ucc allreduce not implemented for type or operation"};
  }

  ucc_coll_req_h req;
  ucc_coll_args_t args;

  args.mask = 0;
  args.coll_type = UCC_COLL_TYPE_ALLREDUCE;
  args.src.info.buffer = const_cast<void *>(send_buf);
  args.src.info.count = count;
  args.src.info.datatype = dt;
  args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
  args.dst.info.buffer = rcv_buf;
  args.dst.info.count = count;
  args.dst.info.datatype = dt;
  args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;
  args.op = op;

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_collective_init(&args, &req, ucc_team_));
  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_collective_post(req));

  while (UCC_INPROGRESS == ucc_collective_test(req)) {
    RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_context_progress(ucc_context_));
  }

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_collective_finalize(req));
  return Status::OK();
}

UccTableGatherImpl::UccTableGatherImpl(ucc_team_h ucc_team,
                                       ucc_context_h ucc_context, int rk, int ws)
    : ucc_team_(ucc_team), ucc_context_(ucc_context), world_size(ws), rank(rk) {}

void UccTableGatherImpl::Init(int32_t num_buffers) {
  this->requests_.resize(num_buffers);
  this->args_.resize(num_buffers);
}

Status UccTableGatherImpl::GatherBufferSizes(const int32_t *send_data, int32_t num_buffers,
                         int32_t *rcv_data, int32_t gather_root) const {
  ucc_coll_args_t args;
  ucc_coll_req_h req;

  args.mask = 0;
  args.coll_type = UCC_COLL_TYPE_GATHER;
  args.root = gather_root;

  args.src.info.buffer = const_cast<int32_t *>(send_data);
  args.src.info.count = static_cast<uint64_t>(num_buffers);
  args.src.info.datatype = UCC_DT_INT32;
  args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

  if(rank == gather_root) {
    args.dst.info.buffer = rcv_data;
    args.dst.info.count = static_cast<uint64_t>(num_buffers * world_size);
    args.dst.info.datatype = UCC_DT_INT32;
    args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;
  }

  RETURN_CYLON_STATUS_IF_UCC_FAILED(
      ucc_collective_init(&args, &req, ucc_team_));

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_collective_post(req));

  ucc_status_t status;

  while (UCC_OK != (status = ucc_collective_test(req))) {
    RETURN_CYLON_STATUS_IF_UCC_FAILED(status);
    RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_context_progress(ucc_context_));
  }

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_collective_finalize(req));
  return Status::OK();
}

Status UccTableGatherImpl::IgatherBufferData(
    int32_t buf_idx, const uint8_t *send_data, int32_t send_count,
    uint8_t *recv_data, const std::vector<int32_t> &recv_count,
    const std::vector<int32_t> &displacements, int32_t gather_root) {
  ucc_coll_args_t &args = args_[buf_idx];

  args.mask = 0;
  args.coll_type = UCC_COLL_TYPE_GATHERV;
  args.root = gather_root;

  args.src.info.buffer = const_cast<uint8_t *>(send_data);
  args.src.info.count = static_cast<uint64_t>(send_count);
  args.src.info.datatype = UCC_DT_UINT8;
  args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

  if(rank == gather_root) {
    args.dst.info_v.buffer = recv_data;

    args.dst.info_v.counts = (ucc_count_t *)recv_count.data();
    args.dst.info_v.displacements =
        (ucc_aint_t *)displacements.data();
    args.dst.info_v.datatype = UCC_DT_UINT8;
    args.dst.info_v.mem_type = UCC_MEMORY_TYPE_HOST;
  }

  RETURN_CYLON_STATUS_IF_UCC_FAILED(
      ucc_collective_init(&args, &requests_[buf_idx], ucc_team_));

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_collective_post(requests_[buf_idx]));
  return Status::OK();
}

Status UccTableGatherImpl::WaitAll(int32_t num_buffers) {
  CYLON_UNUSED(num_buffers);
  RETURN_CYLON_STATUS_IF_UCC_FAILED(WaitAllHelper(requests_, ucc_context_));
  return Status::OK();
}

UccTableGatherImpl::~UccTableGatherImpl() {
  for (auto req : requests_) {
    // ucc_collective_finalize(req);
  }
}

UccTableBcastImpl::UccTableBcastImpl(ucc_team_h ucc_team, ucc_context_h ucc_context,
                  int ws): ucc_team_(ucc_team), ucc_context_(ucc_context) {};

void UccTableBcastImpl::Init(int32_t num_buffers) {
  reqs.resize(num_buffers);
  args.resize(num_buffers);
}

Status UccTableBcastImpl::BcastBufferSizes(int32_t *buffer, int32_t count,
                                           int32_t bcast_root) const {
  ucc_coll_args_t args;
  ucc_coll_req_h req;

  args.mask = 0;
  args.coll_type = UCC_COLL_TYPE_BCAST;
  args.root = bcast_root;

  args.src.info.buffer = buffer;
  args.src.info.count = count;
  args.src.info.datatype = UCC_DT_INT32;
  args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

  RETURN_CYLON_STATUS_IF_UCC_FAILED(
      ucc_collective_init(&args, &req, ucc_team_));

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_collective_post(req));

  ucc_status_t status;

  while (UCC_OK != (status = ucc_collective_test(req))) {
    RETURN_CYLON_STATUS_IF_UCC_FAILED(status);
    RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_context_progress(ucc_context_));
  }

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_collective_finalize(req));

  return Status::OK();
}

Status UccTableBcastImpl::BcastBufferData(uint8_t *buf_data, int32_t send_count,
                                        int32_t bcast_root) const {
  ucc_coll_args_t args;
  ucc_coll_req_h req;

  args.mask = 0;
  args.coll_type = UCC_COLL_TYPE_BCAST;
  args.root = bcast_root;

  args.src.info.buffer = buf_data;
  args.src.info.count = send_count;
  args.src.info.datatype = UCC_DT_UINT8;
  args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

  RETURN_CYLON_STATUS_IF_UCC_FAILED(
      ucc_collective_init(&args, &req, ucc_team_));

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_collective_post(req));

  ucc_status_t status;

  while (UCC_OK != (status = ucc_collective_test(req))) {
    RETURN_CYLON_STATUS_IF_UCC_FAILED(status);
    RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_context_progress(ucc_context_));
  }

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_collective_finalize(req));

  return Status::OK();
}

Status UccTableBcastImpl::IbcastBufferData(int32_t buf_idx, uint8_t *buf_data,
                                         int32_t send_count,
                                         int32_t bcast_root) {
  ucc_coll_args_t& arg = args[buf_idx];
  ucc_coll_req_h& req = reqs[buf_idx];

  arg.mask = 0;
  arg.coll_type = UCC_COLL_TYPE_BCAST;
  arg.root = bcast_root;

  arg.src.info.buffer = buf_data;
  arg.src.info.count = send_count;
  arg.src.info.datatype = UCC_DT_UINT8;
  arg.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

  RETURN_CYLON_STATUS_IF_UCC_FAILED(
      ucc_collective_init(&arg, &req, ucc_team_));

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_collective_post(req));

  return Status::OK();
}

Status UccTableBcastImpl::WaitAll(int32_t num_buffers) {
  CYLON_UNUSED(num_buffers);
  RETURN_CYLON_STATUS_IF_UCC_FAILED(WaitAllHelper(reqs, ucc_context_));
  return Status::OK();
}

UccTableBcastImpl::~UccTableBcastImpl() {
  for (auto req : reqs) {
    ucc_collective_finalize(req);
  }
}

UccAllGatherImpl::UccAllGatherImpl(ucc_team_h ucc_team,
                                   ucc_context_h ucc_context, int ws)
    : ucc_team_(ucc_team), ucc_context_(ucc_context), world_size(ws) {}

Status UccAllGatherImpl::AllgatherBufferSize(const int32_t *send_data,
                                           int32_t num_buffers,
                                           int32_t *rcv_data) const {
  ucc_coll_args_t args;
  ucc_coll_req_h req;

  args.mask = 0;
  args.coll_type = UCC_COLL_TYPE_ALLGATHER;

  args.src.info.buffer = const_cast<int32_t *>(send_data);
  args.src.info.count = static_cast<uint64_t>(num_buffers);
  args.src.info.datatype = UCC_DT_INT32;
  args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
  args.dst.info.buffer = rcv_data;
  args.dst.info.count = static_cast<uint64_t>(num_buffers * world_size);
  args.dst.info.datatype = UCC_DT_INT32;
  args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

  RETURN_CYLON_STATUS_IF_UCC_FAILED(
      ucc_collective_init(&args, &req, ucc_team_));
  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_collective_post(req));

  ucc_status_t status;

  while (UCC_OK != (status = ucc_collective_test(req))) {
    RETURN_CYLON_STATUS_IF_UCC_FAILED(status);
    RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_context_progress(ucc_context_));
  }

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_collective_finalize(req));

  return Status::OK();
}

Status UccAllGatherImpl::IallgatherBufferData(int32_t buf_idx, const uint8_t *send_data,
                            int32_t send_count, uint8_t *recv_data,
                            const std::vector<int32_t> &recv_count,
                            const std::vector<int32_t> &displacements) {
  requests_.resize(3);
  args_.resize(3);

  ucc_coll_args_t &args = args_[buf_idx];

  args.mask = 0;
  args.coll_type = UCC_COLL_TYPE_ALLGATHERV;

  args.src.info.buffer = const_cast<uint8_t *>(send_data);
  args.src.info.count = static_cast<uint64_t>(send_count);
  args.src.info.datatype = UCC_DT_UINT8;
  args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

  args.dst.info_v.buffer = recv_data;
  args.dst.info_v.counts = (ucc_count_t *)recv_count.data();
  args.dst.info_v.displacements = (ucc_aint_t *)displacements.data();
  args.dst.info_v.datatype = UCC_DT_UINT8;
  args.dst.info_v.mem_type = UCC_MEMORY_TYPE_HOST;

  RETURN_CYLON_STATUS_IF_UCC_FAILED(
      ucc_collective_init(&args, &requests_[buf_idx], ucc_team_));

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_collective_post(requests_[buf_idx]));
  return Status::OK();
}

Status UccAllGatherImpl::WaitAll() {
  RETURN_CYLON_STATUS_IF_UCC_FAILED(WaitAllHelper(requests_, ucc_context_));
  return Status::OK();
}

UccAllGatherImpl::~UccAllGatherImpl() {
  for (auto req : requests_) {
    ucc_collective_finalize(req);
  }
}
}  // namespace ucc
}  // namespace cylon
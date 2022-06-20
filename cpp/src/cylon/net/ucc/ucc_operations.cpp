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
  // RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Iallgatherv(
  //     send_data, send_count, MPI_UINT8_T, recv_data, recv_count.data(),
  //     displacements.data(), MPI_UINT8_T, comm_, &requests_[buf_idx]));
  
  ucc_coll_args_t &args = args_[buf_idx];

  args.mask = 0;
  args.coll_type = UCC_COLL_TYPE_ALLGATHERV;

  args.src.info.buffer = const_cast<uint8_t *>(send_data);
  args.src.info.count = static_cast<uint64_t>(send_count);
  args.src.info.datatype = UCC_DT_UINT8;
  args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

  args.dst.info_v.buffer = recv_data;

  counts_[buf_idx].insert(counts_[buf_idx].end(), recv_count.begin(), recv_count.end());
  displacements_[buf_idx].insert(displacements_[buf_idx].end(), displacements.begin(),
                          displacements.end());

  // std::vector<uint64_t> displacements_v(displacements.begin(), displacements.end());

  args.dst.info_v.counts = counts_[buf_idx].data();
  args.dst.info_v.displacements = displacements_[buf_idx].data();
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
  // RETURN_CYLON_STATUS_IF_MPI_FAILED(
  //     MPI_Waitall(num_buffers, requests_.data(), statuses_.data()));
  // ucc_status_t status;

  // TODO: adopt ucc test `waitall`'s algorithm
  // for(int i = 0; i < num_buffers; i++) {
  //   while (UCC_OK != (status = ucc_collective_test(requests_[i]))) {
  //     RETURN_CYLON_STATUS_IF_UCC_FAILED(status);
  //     // std::cout<<"status: "<<status<<std::endl;
  //     RETURN_CYLON_STATUS_IF_UCC_FAILED(status = ucc_context_progress(ucc_context_));
  //   }
  // }

  RETURN_CYLON_STATUS_IF_UCC_FAILED(WaitAllHelper(requests_, ucc_context_));

  return Status::OK();
}

UccTableAllgatherImpl::UccTableAllgatherImpl(ucc_team_h ucc_team,
                                             ucc_context_h ucc_context,
                                             int world_sz)
    : TableAllgatherImpl(),
      ucc_team_(ucc_team),
      ucc_context_(ucc_context),
      requests_({}),
      args_({}),
      counts_({}),
      displacements_({}),
      world_size(world_sz){}

void UccTableAllgatherImpl::Init(int num_buffers) {
  requests_.resize(num_buffers);
  args_.resize(num_buffers);
  counts_.resize(num_buffers);
  displacements_.resize(num_buffers);
}

UccTableAllgatherImpl::~UccTableAllgatherImpl() {
  for (auto req: requests_) {
    ucc_collective_finalize(req);
  }
}

}  // namespace ucc
}  // namespace cylon
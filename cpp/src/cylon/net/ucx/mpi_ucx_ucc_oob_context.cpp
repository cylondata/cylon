#include <cylon/net/ucx/mpi_ucx_ucc_oob_context.hpp>

namespace cylon {
namespace net {


Status UCXMPIOOBContext::InitOOB() {
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Init(nullptr, nullptr));
  }
  return Status::OK();
}

Status UCXMPIOOBContext::getWorldSizeAndRank(int &world_size, int &rank) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  return Status::OK();
}

Status UCXMPIOOBContext::OOBAllgather(uint8_t *src, uint8_t *dst,
                                      size_t srcSize, size_t dstSize) {
  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Allgather(
      src, srcSize, MPI_BYTE, dst, dstSize, MPI_BYTE, MPI_COMM_WORLD));
  return Status::OK();
}

Status UCXMPIOOBContext::Finalize() {
  int mpi_finalized;
  MPI_Finalized(&mpi_finalized);
  if (!mpi_finalized) {
    MPI_Finalize();
  }
  return Status::OK();
}

#ifdef BUILD_CYLON_UCC


void UCCMPIOOBContext::InitOOB(int rank){
  CYLON_UNUSED(rank);
};

std::shared_ptr<UCXOOBContext> UCCMPIOOBContext::makeUCXOOBContext() {
  return std::make_shared<UCXMPIOOBContext>();
}

OOBType UCCMPIOOBContext::Type() { return OOBType::OOB_MPI; }

void *UCCMPIOOBContext::getCollInfo() {
    return reinterpret_cast<void*>(MPI_COMM_WORLD);
}

ucc_status_t UCCMPIOOBContext::oob_allgather(void *sbuf, void *rbuf,
                                             size_t msglen, void *coll_info,
                                             void **req) {
  auto comm = reinterpret_cast<MPI_Comm>(coll_info);
  MPI_Request request;

  MPI_Iallgather(sbuf, (int)msglen, MPI_BYTE, rbuf, (int)msglen, MPI_BYTE, comm,
                 &request);
  *req = reinterpret_cast<void *>(request);
  return UCC_OK;
}

ucc_status_t UCCMPIOOBContext::oob_allgather_test(void *req) {
  auto request = (MPI_Request)req;
  int completed;

  MPI_Test(&request, &completed, MPI_STATUS_IGNORE);
  return completed ? UCC_OK : UCC_INPROGRESS;
}

ucc_status_t UCCMPIOOBContext::oob_allgather_free(void *req) {
  (void)req;
  return UCC_OK;
}

        Status UCCMPIOOBContext::Finalize() {
            return Status::OK();
        }

#endif
}  // namespace net
}  // namespace cylon
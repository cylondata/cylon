#include <cylon/net/ucx/ucx_ucc_oob_contexts.hpp>

namespace cylon {
namespace net {
UCXRedisOOBContext::UCXRedisOOBContext(std::shared_ptr<sw::redis::Redis> rds,
                                       int ws)
    : redis(rds), world_size(ws) {}

Status UCXRedisOOBContext::InitOOB() { return Status::OK(); };

Status UCXRedisOOBContext::getWorldSizeAndRank(int &world_size, int &rank) {
  world_size = this->world_size;
  int num_cur_processes = redis->incr("num_cur_processes");
  rank = this->rank = num_cur_processes - 1;

  return Status::OK();
}

Status UCXRedisOOBContext::OOBAllgather(uint8_t *src, uint8_t *dst,
                                        size_t srcSize, size_t dstSize) {
  const auto ucc_worker_addr_mp_str = "ucp_worker_addr_mp";
  redis->hset(ucc_worker_addr_mp_str, std::to_string(rank),
              std::string((char *)src, (char *)src + srcSize));
  std::vector<int> v(world_size, 0);
  redis->lpush("ucx_helper" + std::to_string(rank), v.begin(), v.end());

  for (int i = 0; i < world_size; i++) {
    if (i == rank) continue;
    auto i_str = std::to_string(i);
    auto helperName = "ucx_helper" + i_str;

    auto val = redis->hget(ucc_worker_addr_mp_str, i_str);
    while (!val) {
      redis->blpop(helperName);
      val = redis->hget(ucc_worker_addr_mp_str, i_str);
    }

    memcpy(dst + i * srcSize, val.value().data(), srcSize);
  }

  return Status::OK();
}

Status UCXRedisOOBContext::Finalize() { return Status::OK(); };

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
void UCCRedisOOBContext::InitOOB(int rank) { this->rank = rank; }

std::shared_ptr<UCXOOBContext> UCCRedisOOBContext::makeUCXOOBContext() {
  return std::make_shared<UCXRedisOOBContext>(redis, world_size);
}

void *UCCRedisOOBContext::getCollInfo() { return this; }

ucc_status_t UCCRedisOOBContext::oob_allgather(void *sbuf, void *rbuf,
                                               size_t msglen, void *coll_info,
                                               void **req) {
  auto oob_allgather_func = [](void *sbuf, void *rbuf, size_t msglen,
                               void *coll_info, void **req) { return UCC_OK; };
  int world_size = ((UCCRedisOOBContext *)coll_info)->world_size;
  int rank = ((UCCRedisOOBContext *)coll_info)->rank;
  int num_comm = ((UCCRedisOOBContext *)coll_info)->num_oob_allgather;
  ((UCCRedisOOBContext *)coll_info)->num_oob_allgather++;

  auto &redis = ((UCCRedisOOBContext *)coll_info)->redis;
  *req = rbuf;
  std::string s((char *)sbuf, ((char *)sbuf) + msglen);

  redis->hset("ucc_oob_mp" + std::to_string(num_comm), std::to_string(rank), s);
  redis->lpush(
      "ucc_helper" + std::to_string(num_comm) + ":" + std::to_string(rank),
      "0");

  for (int i = 0; i < world_size; i++) {
    if (i == rank) {
      memcpy(rbuf + i * msglen, s.data(), msglen);
    } else {
      auto helperName =
          "ucc_helper" + std::to_string(num_comm) + ":" + std::to_string(i);

      // val = redis.hget("ucp_worker_addr_mp", std::to_string(i));
      sw::redis::OptionalString val;
      do {
        redis->brpoplpush(helperName, helperName, 0);
        val = redis->hget("ucc_oob_mp" + std::to_string(num_comm),
                          std::to_string(i));
      } while (!val);

      memcpy(rbuf + i * msglen, val.value().data(), msglen);
    }
  }

  // maybe need to do some cleanups here

  return UCC_OK;
}

UCCRedisOOBContext::UCCRedisOOBContext(int ws,
                                       std::shared_ptr<sw::redis::Redis> &rds)
    : world_size(ws), redis(rds) {}

ucc_status_t UCCRedisOOBContext::oob_allgather_test(void *req) {
  CYLON_UNUSED(req);
  return UCC_OK;
}

ucc_status_t UCCRedisOOBContext::oob_allgather_free(void *req) {
  CYLON_UNUSED(req);
  return UCC_OK;
}

OOBType UCCRedisOOBContext::Type() { return OOBType::OOB_REDIS; }

std::shared_ptr<sw::redis::Redis> UCCRedisOOBContext::getRedis() {
  return this->redis;
}

int UCCRedisOOBContext::getWorldSize() { return world_size; }

void UCCRedisOOBContext::setRank(int rk) { rank = rk; }

int UCCRedisOOBContext::getRank() { return rank; }

void UCCMPIOOBContext::InitOOB(int rank){};

std::shared_ptr<UCXOOBContext> UCCMPIOOBContext::makeUCXOOBContext() {
  return std::make_shared<UCXMPIOOBContext>();
}

OOBType UCCMPIOOBContext::Type() { return OOBType::OOB_MPI; }

void *UCCMPIOOBContext::getCollInfo() { return MPI_COMM_WORLD; }

ucc_status_t UCCMPIOOBContext::oob_allgather(void *sbuf, void *rbuf,
                                             size_t msglen, void *coll_info,
                                             void **req) {
  auto comm = (MPI_Comm)coll_info;
  MPI_Request request;

  MPI_Iallgather(sbuf, (int)msglen, MPI_BYTE, rbuf, (int)msglen, MPI_BYTE, comm,
                 &request);
  *req = (void *)request;
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
#endif
}  // namespace net
}  // namespace cylon
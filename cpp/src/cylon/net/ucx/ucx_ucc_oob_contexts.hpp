#ifndef CYLON_SRC_CYLON_COMM_UCXUCCOOBCONTEXTS_H_
#define CYLON_SRC_CYLON_COMM_UCXUCCOOBCONTEXTS_H_
#include <cylon/net/comm_config.hpp>
#include <cylon/net/communicator.hpp>
#include <cylon/net/ucx/ucx_operations.hpp>

#include "cylon/util/macros.hpp"
#include "sw/redis++/redis++.h"

#ifdef BUILD_CYLON_UCC
#include <ucc/api/ucc.h>
#endif

namespace cylon {
namespace net {
enum OOBType { OOB_MPI = 0, OOB_REDIS = 1 };

class UCXOOBContext {
 public:
  virtual Status InitOOB() = 0;
  virtual Status getWorldSizeAndRank(int &world_size, int &rank) = 0;
  virtual Status OOBAllgather(uint8_t *src, uint8_t *dst, size_t srcSize,
                              size_t dstSize) = 0;
  virtual Status Finalize() = 0;
};

class UCXRedisOOBContext : public UCXOOBContext {
 public:
  UCXRedisOOBContext(std::shared_ptr<sw::redis::Redis> redis, int world_size);
  Status InitOOB() override;

  Status getWorldSizeAndRank(int &world_size, int &rank) override;

  Status OOBAllgather(uint8_t *src, uint8_t *dst, size_t srcSize,
                      size_t dstSize) override;

  Status Finalize();

 private:
  std::shared_ptr<sw::redis::Redis> redis;
  int world_size;
  int rank = -1;
};

class UCXMPIOOBContext : public UCXOOBContext {
 public:
  Status InitOOB() override;

  Status getWorldSizeAndRank(int &world_size, int &rank);

  Status OOBAllgather(uint8_t *src, uint8_t *dst, size_t srcSize,
                      size_t dstSize) override;

  Status Finalize() override;
};

#ifdef BUILD_CYLON_UCC
class UCCOOBContext {
 public:
  virtual OOBType Type() = 0;
  virtual std::shared_ptr<UCXOOBContext> makeUCXOOBContext() = 0;
  virtual void InitOOB(int rank) = 0;
  virtual void *getCollInfo() = 0;
};

class UCCRedisOOBContext : public UCCOOBContext {
 public:
  void InitOOB(int rank) override;

  std::shared_ptr<UCXOOBContext> makeUCXOOBContext() override;

  void *getCollInfo() override;

  OOBType Type() override;

  UCCRedisOOBContext(int world_size, std::shared_ptr<sw::redis::Redis> &redis);

  static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                    void *coll_info, void **req);
  static ucc_status_t oob_allgather_test(void *req);
  static ucc_status_t oob_allgather_free(void *req);

  std::shared_ptr<sw::redis::Redis> getRedis();
  int getWorldSize();
  void setRank(int rk);
  int getRank();

 private:
  int world_size;
  int rank = -1;
  std::shared_ptr<sw::redis::Redis> redis;
  int num_oob_allgather = 0;
};

class UCCMPIOOBContext : public UCCOOBContext {
 public:
  UCCMPIOOBContext() = default;
  void InitOOB(int rank) override;
  std::shared_ptr<UCXOOBContext> makeUCXOOBContext() override;

  OOBType Type() override;

  void *getCollInfo();

  static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                    void *coll_info, void **req);

  static ucc_status_t oob_allgather_test(void *req);

  static ucc_status_t oob_allgather_free(void *req);
};
#endif
}  // namespace net
}  // namespace cylon
#endif
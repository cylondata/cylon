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

#ifndef CYLON_SRC_CYLON_COMM_UCXCOMMUNICATOR_H_
#define CYLON_SRC_CYLON_COMM_UCXCOMMUNICATOR_H_

#include <cylon/net/comm_config.hpp>
#include <cylon/net/communicator.hpp>
#include <cylon/net/ucx/ucx_operations.hpp>

#include "cylon/util/macros.hpp"
#ifdef BUILD_CYLON_UCC
#include <ucc/api/ucc.h>
#include "sw/redis++/redis++.h"
#endif

namespace cylon {
namespace net {

enum OOBType { OOB_MPI = 0, OOB_REDIS = 1 };

class UCXOOBContext {
public:
  virtual Status InitOOB() = 0;
  virtual Status getWorldSizeAndRank(int &world_size, int &rank) = 0;
  virtual Status OOBAllgather(uint8_t *src, uint8_t *dst, size_t srcSize, size_t dstSize) = 0;
  virtual Status Finalize() = 0;
};

class UCXRedisOOBContext : public UCXOOBContext {
public:
  UCXRedisOOBContext(std::shared_ptr<sw::redis::Redis> redis, int world_size) {
    this->world_size = world_size;
    this->redis = redis;
  }
  Status InitOOB() override {
    return Status::OK();
  };

  Status getWorldSizeAndRank(int &world_size, int &rank) override {
    world_size = this->world_size;
    int num_cur_processes = redis->incr("num_cur_processes");
    rank = this->rank = num_cur_processes - 1;

    return Status::OK();
  }

  Status OOBAllgather(uint8_t *src, uint8_t *dst, size_t srcSize, size_t dstSize) override {
    redis->hset("ucp_worker_addr_mp", std::to_string(rank), std::string((char*)src, (char*)src + srcSize));
    std::vector<int> v(world_size, 0);
    redis->lpush("ucx_helper" + std::to_string(rank), v.begin(), v.end());

    for(int i = 0; i < world_size; i++) {
      if (i == rank) continue;
      auto helperName = "ucx_helper" + std::to_string(i);

      auto val = redis->hget("ucp_worker_addr_mp", std::to_string(i));
      while (!val) {
        redis->blpop(helperName);
        val = redis->hget("ucp_worker_addr_mp", std::to_string(i));
      }

      memcpy(dst + i * srcSize, val.value().data(), srcSize);
    }

    return Status::OK();
  }

  Status Finalize() override {
    return Status::OK();
  };

private:
  std::shared_ptr<sw::redis::Redis> redis;
  int world_size;
  int rank = -1;
};

class UCXMPIOOBContext : public UCXOOBContext {
 public:
  Status InitOOB() override {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
      RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Init(nullptr, nullptr));
    }
    return Status::OK();
  }

  Status getWorldSizeAndRank(int &world_size, int &rank) override {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    return Status::OK();
  }

  Status OOBAllgather(uint8_t *src, uint8_t *dst, size_t srcSize, size_t dstSize) override {
    RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Allgather(
        src, srcSize, MPI_BYTE,
        dst, dstSize, MPI_BYTE,
        MPI_COMM_WORLD));
    return Status::OK();
  }

  Status Finalize() override {
    int mpi_finalized;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized) {
      MPI_Finalize();
    }
    return Status::OK();
  }
};

class UCXConfig : public CommConfig {
  CommType Type() override {
    return CommType::UCX;
  }

 public:
  explicit UCXConfig(std::shared_ptr<UCXOOBContext> oobContext) {
    this->oobContext = oobContext;
  }

  static std::shared_ptr<UCXConfig> Make(
      std::shared_ptr<UCXOOBContext> oobContext) {
    return std::make_shared<UCXConfig>(oobContext);
  }

  void setOOBContext(std::shared_ptr<UCXOOBContext> oobContext) {
    this->oobContext = oobContext;
  }

  std::shared_ptr<UCXOOBContext> getOOBContext() { return this->oobContext; }

 private:
  std::shared_ptr<UCXOOBContext> oobContext;
};

class UCCOOBContext {
public:
 virtual OOBType Type() = 0;
 virtual std::shared_ptr<UCXOOBContext> makeUCXOOBContext() = 0;
 virtual void InitOOB(int rank) = 0;
 virtual void* getCollInfo() = 0;
};

typedef ucc_status_t oob_func_t(void *sbuf, void *rbuf, size_t msglen,
                                          void *coll_info, void **req);
typedef ucc_status_t oob_test_func_t(void* req);
typedef ucc_status_t oob_free_func_t(void* req);

class UCCRedisOOBContext : public UCCOOBContext {
 public:
  void InitOOB(int rank) override {
    this->rank = rank;
  }

  std::shared_ptr<UCXOOBContext> makeUCXOOBContext() override {
    return std::make_shared<UCXRedisOOBContext>(redis, world_size);
  }

  void* getCollInfo() override {
    return this;
  }

  OOBType Type() override;

  UCCRedisOOBContext(int world_size, std::shared_ptr<sw::redis::Redis>& redis);

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
  void InitOOB(int rank) override{};
  std::shared_ptr<UCXOOBContext> makeUCXOOBContext() override {
    return std::make_shared<UCXMPIOOBContext>();
  }

  OOBType Type() override {
    return OOBType::OOB_MPI;
  }

  void* getCollInfo() {
    return MPI_COMM_WORLD;
  }

  static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                  void *coll_info, void **req) {
    auto comm = (MPI_Comm) coll_info;
    MPI_Request request;

    MPI_Iallgather(sbuf, (int) msglen, MPI_BYTE, rbuf, (int) msglen, MPI_BYTE, comm,
                  &request);
    *req = (void *) request;
    return UCC_OK;
  }

  static ucc_status_t oob_allgather_test(void *req) {
    auto request = (MPI_Request) req;
    int completed;

    MPI_Test(&request, &completed, MPI_STATUS_IGNORE);
    return completed ? UCC_OK : UCC_INPROGRESS;
  }

  static ucc_status_t oob_allgather_free(void *req) {
    (void) req;
    return UCC_OK;
  }
};

  // should carry resources universal to UCC
class UCCConfig : public CommConfig {
    CommType Type() override;

   public:
    explicit UCCConfig(std::shared_ptr<UCCOOBContext> oobContext);
    static std::shared_ptr<UCCConfig> Make(
        std::shared_ptr<UCCOOBContext> &oobContext);

    // TODO: hide these
    std::shared_ptr<UCCOOBContext> oobContext;
};

class UCXCommunicator : public Communicator {
 public:
  explicit UCXCommunicator(MemoryPool *pool);

  ~UCXCommunicator() override = default;

  std::unique_ptr<Channel> CreateChannel() const override;
  int GetRank() const override;
  int GetWorldSize() const override;
  void Finalize() override;
  void Barrier() override;
  CommType GetCommType() const override;

  Status AllGather(const std::shared_ptr<Table> &table,
                   std::vector<std::shared_ptr<Table>> *out) const override;
  Status Gather(const std::shared_ptr<Table> &table,
                int gather_root,
                bool gather_from_root,
                std::vector<std::shared_ptr<Table>> *out) const override;
  Status Bcast(std::shared_ptr<Table> *table, int bcast_root,
               const std::shared_ptr<CylonContext> &ctx) const override;
  Status AllReduce(const std::shared_ptr<Column> &column,
                   net::ReduceOp reduce_op,
                   std::shared_ptr<Column> *output) const override;
  Status AllReduce(const std::shared_ptr<Scalar> &values,
                   net::ReduceOp reduce_op,
                   std::shared_ptr<Scalar> *output) const override;
  Status Allgather(const std::shared_ptr<Column> &values,
                   std::vector<std::shared_ptr<Column>> *output) const override;
  Status Allgather(const std::shared_ptr<Scalar> &value,
                   std::shared_ptr<Column> *output) const override;

  static Status Make(const std::shared_ptr<CommConfig> &config, MemoryPool *pool,
                     std::shared_ptr<Communicator> *out);

  static Status MakeWithMPI(const std::shared_ptr<CommConfig> &config,
                     MemoryPool *pool, std::shared_ptr<Communicator> *out);

  static Status MakeWithRedis(const std::shared_ptr<CommConfig> &config,
                            MemoryPool *pool,
                            std::shared_ptr<Communicator> *out);

  // # UCX specific attributes - These need to be passed to the channels created from the communicator
  // The worker for receiving
  ucp_worker_h ucpRecvWorker{};
  // The worker for sending
  ucp_worker_h ucpSendWorker{};
  // Endpoint Map
  std::unordered_map<int, ucp_ep_h> endPointMap;
  // UCP Context - Holds a UCP communication instance's global information.
  ucp_context_h ucpContext{};

  std::shared_ptr<UCXOOBContext> oobContext;
};

#ifdef BUILD_CYLON_UCC
class UCXUCCCommunicator: public Communicator{
 public:
  explicit UCXUCCCommunicator(std::shared_ptr<Communicator> ucx_comm,
                              std::shared_ptr<UCCOOBContext>& oobContext);

  static Status Make(const std::shared_ptr<CommConfig> &config,
                     MemoryPool *pool, std::shared_ptr<Communicator> *out);

  CommType GetCommType() const override;
  std::unique_ptr<Channel> CreateChannel() const override;
  void Finalize() override;
  void Barrier() override;
  Status AllGather(const std::shared_ptr<Table> &table,
                   std::vector<std::shared_ptr<Table>> *out) const override;
  Status Gather(const std::shared_ptr<Table> &table,
                int gather_root,
                bool gather_from_root,
                std::vector<std::shared_ptr<Table>> *out) const override;
  Status Bcast(std::shared_ptr<Table> *table,
               int bcast_root,
               const std::shared_ptr<CylonContext> &ctx) const override;
  Status AllReduce(const std::shared_ptr<Column> &values,
                   net::ReduceOp reduce_op,
                   std::shared_ptr<Column> *output) const override;
  Status Allgather(const std::shared_ptr<Column> &values,
                   std::vector<std::shared_ptr<Column>> *output) const override;
  Status AllReduce(const std::shared_ptr<Scalar> &value,
                   net::ReduceOp reduce_op,
                   std::shared_ptr<Scalar> *output) const override;
  Status Allgather(const std::shared_ptr<Scalar> &value,
                   std::shared_ptr<Column> *output) const override;

  ucc_team_h uccTeam{};
  ucc_context_h uccContext{};
  std::shared_ptr<Communicator> ucx_comm_;
  std::shared_ptr<UCCOOBContext> oobContext;
};
#endif
}
}
#endif //CYLON_SRC_CYLON_COMM_UCXCOMMUNICATOR_H_

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
#include <glog/logging.h>

#include "cylon/net/communicator.hpp"
#include "cylon/net/ucx/ucx_communicator.hpp"
#include "cylon/net/ucx/ucx_channel.hpp"
#include "cylon/util/macros.hpp"

#ifdef BUILD_CYLON_UCC
#include "cylon/net/ucc/ucc_operations.hpp"
#include "sw/redis++/redis++.h"
#endif

namespace cylon {
namespace net {

void mpi_check_and_finalize() {
  int mpi_finalized;
  MPI_Finalized(&mpi_finalized);
  if (!mpi_finalized) {
    MPI_Finalize();
  }
}

CommType UCXConfig::Type() {
  return CommType::UCX;
}

std::shared_ptr<UCXConfig> UCXConfig::Make() {
  return std::make_shared<UCXConfig>();
}

std::unique_ptr<Channel> UCXCommunicator::CreateChannel() const {
  return std::make_unique<UCXChannel>(this);
}

int UCXCommunicator::GetRank() const {
  return this->rank;
}
int UCXCommunicator::GetWorldSize() const {
  return this->world_size;
}

Status UCXCommunicator::AllGather(const std::shared_ptr<Table> &table,
                                  std::vector<std::shared_ptr<Table>> *out) const {
  CYLON_UNUSED(table);
  CYLON_UNUSED(out);
  return {Code::NotImplemented, "All gather not implemented for ucx"};
}

Status UCXCommunicator::Gather(const std::shared_ptr<Table> &table,
                               int gather_root,
                               bool gather_from_root,
                               std::vector<std::shared_ptr<Table>> *out) const {
  CYLON_UNUSED(table);
  CYLON_UNUSED(gather_root);
  CYLON_UNUSED(gather_from_root);
  CYLON_UNUSED(out);
  return {Code::NotImplemented, "All gather not implemented for ucx"};
}

Status UCXCommunicator::Bcast(std::shared_ptr<Table> *table, int bcast_root,
                              const std::shared_ptr<CylonContext> &ctx) const {
  CYLON_UNUSED(table);
  CYLON_UNUSED(bcast_root);
  CYLON_UNUSED(ctx);
  return {Code::NotImplemented, "Bcast not implemented for ucx"};
}

Status UCXCommunicator::AllReduce(const std::shared_ptr<Column> &column,
                                  net::ReduceOp reduce_op,
                                  std::shared_ptr<Column> *output) const {
  CYLON_UNUSED(column);
  CYLON_UNUSED(reduce_op);
  CYLON_UNUSED(output);
  return {Code::NotImplemented, "Allreduce not implemented for ucx"};
}

UCXCommunicator::UCXCommunicator(MemoryPool *pool) : Communicator(pool, -1, -1) {}

Status UCXCommunicator::AllReduce(const std::shared_ptr<Scalar> &values,
                                  net::ReduceOp reduce_op,
                                  std::shared_ptr<Scalar> *output) const {
  CYLON_UNUSED(values);
  CYLON_UNUSED(reduce_op);
  CYLON_UNUSED(output);
  return {Code::NotImplemented, "Allreduce not implemented for ucx"};
}

Status UCXCommunicator::Allgather(const std::shared_ptr<Column> &values,
                                  std::vector<std::shared_ptr<Column>> *output) const {
  CYLON_UNUSED(values);
  CYLON_UNUSED(output);
  return {Code::NotImplemented, "Allgather not implemented for ucx"};
}

Status UCXCommunicator::Allgather(const std::shared_ptr<Scalar> &value,
                                  std::shared_ptr<Column> *output) const {
  CYLON_UNUSED(value);
  CYLON_UNUSED(output);
  return {Code::NotImplemented, "Allgather not implemented for ucx"};
}

Status UCXCommunicator::Make(const std::shared_ptr<CommConfig> &config, MemoryPool *pool,
                             std::shared_ptr<Communicator> *out) {
  CYLON_UNUSED(config);
  *out = std::make_shared<UCXCommunicator>(pool);
  auto &comm = static_cast<UCXCommunicator &>(**out);
  // Check init functions
  int initialized;
  // Int variable used when iterating
  int sIndx;
  // Address of the UCP Worker for receiving
  cylon::ucx::ucxWorkerAddr *ucpRecvWorkerAddr;
  // Address of the UCP Worker for sending
  cylon::ucx::ucxWorkerAddr *ucpSendWorkerAddr;

  // Status check when creating end-points
  ucs_status_t ucxStatus;
  // Variable to hold the current ucp address
  ucp_address_t *address;

  // MPI init
  MPI_Initialized(&initialized);
  if (!initialized) {
    RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Init(nullptr, nullptr));
  }

  auto redis = sw::redis::Redis("tcp://127.0.0.1:6379");

  auto val = redis.get("world_size");  // val is of type OptionalString. See 'API
                                // Reference' section for details.
                                
  if (val) {
    comm.world_size = std::atoi(val.value().c_str()); // set to environmental variable later
  }  // else key doesn't exist.

  int num_cur_processes = redis.incr("num_cur_processes");
  comm.rank = num_cur_processes - 1;

  // Get the rank for checking send to self, and initializations
  int a, b;
  MPI_Comm_rank(MPI_COMM_WORLD, &a);
  MPI_Comm_size(MPI_COMM_WORLD, &b);
  comm.rank = a;
  comm.world_size = b;

  // comm.rank = a;

  int rank = comm.rank, world_size = comm.world_size;

  // Init context
  RETURN_CYLON_STATUS_IF_FAILED(cylon::ucx::initContext(&comm.ucpContext, nullptr));

  // Init recv worker and get address
  ucpRecvWorkerAddr = cylon::ucx::initWorker(comm.ucpContext, &comm.ucpRecvWorker);
  // Init send worker
  ucpSendWorkerAddr = cylon::ucx::initWorker(comm.ucpContext, &comm.ucpSendWorker);

  // Gather all worker addresses
  // All addresses buffer for allGather
  auto addr_str = std::string((char *)ucpRecvWorkerAddr->addr,
                              ((char *)ucpRecvWorkerAddr->addr) + ucpRecvWorkerAddr->addrSize);

  redis.hset("ucp_worker_addr_mp", std::to_string(comm.rank), addr_str);
  std::vector<int> v(world_size, 0);
  redis.lpush("ucx_helper" + std::to_string(comm.rank), v.begin(), v.end());

  auto allAddresses =
      std::make_unique<char[]>(ucpRecvWorkerAddr->addrSize * world_size);

  for(int i = 0; i < world_size; i++) {
    if (i == rank) continue;
    auto helperName = "ucx_helper" + std::to_string(i);

    val = redis.hget("ucp_worker_addr_mp", std::to_string(i));
    while (!val) {
      redis.blpop(helperName);
      val = redis.hget("ucp_worker_addr_mp", std::to_string(i));
    }

    memcpy(allAddresses.get() + i * ucpRecvWorkerAddr->addrSize,
           val.value().data(), ucpRecvWorkerAddr->addrSize);
  }

  // // Iterate and set the sends
  comm.endPointMap.reserve(world_size);
  for (sIndx = 0; sIndx < world_size; sIndx++) {
    ucp_ep_params_t epParams;
    ucp_ep_h ep;

    // If not self, then check if the worker address has been received.
    //  If self,then assign local worker
    if (rank != sIndx) {
      address = reinterpret_cast<ucp_address_t *>(allAddresses.get()
          + sIndx * ucpRecvWorkerAddr->addrSize);
    } else {
      address = ucpRecvWorkerAddr->addr;
    }

    // Set params for the endpoint
    epParams.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
        UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    epParams.address = address;
    epParams.err_mode = UCP_ERR_HANDLING_MODE_NONE;

    // Create an endpoint
    ucxStatus = ucp_ep_create(comm.ucpSendWorker, &epParams, &ep);

    comm.endPointMap[sIndx] = ep;
    // Check if the endpoint was created properly
    if (ucxStatus != UCS_OK) {
      LOG(FATAL) << "Error when creating the endpoint.";
      return {Code::ExecutionError,
              "Error when creating the endpoint: " + std::string(ucs_status_string(ucxStatus))};
    }
  }

  // Cleanup
  delete (ucpRecvWorkerAddr);
  delete (ucpSendWorkerAddr);

  std::cout<<"a"<<rank<<std::endl;

  return Status::OK();
}

void UCXCommunicator::Finalize() {
  if (!this->IsFinalized()) {
    ucp_cleanup(ucpContext);
    mpi_check_and_finalize();
    finalized = true;
  }
}

void UCXCommunicator::Barrier() {
  MPI_Barrier(MPI_COMM_WORLD);
}

CommType UCXCommunicator::GetCommType() const {
  return UCX;
}

#ifdef BUILD_CYLON_UCC

static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                  void *coll_info, void **req) {
  std::cout<<msglen<<std::endl;
  auto comm = (MPI_Comm) coll_info;
  MPI_Request request;

  MPI_Iallgather(sbuf, (int) msglen, MPI_BYTE, rbuf, (int) msglen, MPI_BYTE, comm,
                 &request);
  *req = (void *) request;
  return UCC_OK;
}

struct redisAllgatherInfo {
  int num_comm;
  int world_size;
  int rank;
  sw::redis::Redis* redis;
};

static ucc_status_t redis_allgather(void *sbuf, void *rbuf, size_t msglen,
                                  void *coll_info, void **req) {

  int world_size = ((redisAllgatherInfo*) coll_info)->world_size;
  int rank = ((redisAllgatherInfo*) coll_info)->rank;

                                  std::cout<<msglen<<" "<<rank<<std::endl;
  int num_comm = ((redisAllgatherInfo*) coll_info)->num_comm;
  ((redisAllgatherInfo*) coll_info)->num_comm ++;

  sw::redis::Redis* redis = ((redisAllgatherInfo*) coll_info)->redis;
  *req = rbuf;
  std::string s((char*)sbuf, ((char*)sbuf) + msglen);

  redis->hset("ucc_oob_mp" + std::to_string(num_comm), std::to_string(rank), s);
  redis->lpush("ucc_helper" + std::to_string(num_comm) + ":" + std::to_string(rank),
               "0");

  MPI_Request request;

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
static ucc_status_t redis_allgather_test(void *req) {
  CYLON_UNUSED(req);
  return UCC_OK;
}

static ucc_status_t oob_allgather_test(void *req) {
  auto request = (MPI_Request) req;
  int completed;

  MPI_Test(&request, &completed, MPI_STATUS_IGNORE);
  return completed ? UCC_OK : UCC_INPROGRESS;
}

static ucc_status_t oob_allgather_free(void *req) {
  CYLON_UNUSED(req);
  return UCC_OK;
}

UCXUCCCommunicator::UCXUCCCommunicator(std::shared_ptr<Communicator> ucx_comm)
    : Communicator(ucx_comm->GetMemoryPool(), ucx_comm->GetRank(), ucx_comm->GetWorldSize()),
      ucx_comm_(std::move(ucx_comm)) {}

Status UCXUCCCommunicator::Make(const std::shared_ptr<CommConfig> &config,
                                MemoryPool *pool,
                                std::shared_ptr<Communicator> *out) {
  std::shared_ptr<Communicator> ucx_comm;
  RETURN_CYLON_STATUS_IF_FAILED(UCXCommunicator::Make(config, pool, &ucx_comm));

  *out = std::make_shared<UCXUCCCommunicator>(std::move(ucx_comm));
  auto &comm = *std::static_pointer_cast<UCXUCCCommunicator>(*out);

  auto redis = sw::redis::Redis("tcp://127.0.0.1:6379");
  redisAllgatherInfo coll_info = {0, comm.GetWorldSize(), comm.GetRank(), &redis};

  // initialize UCC team and context
  ucc_context_params_t ctx_params;
  ucc_team_params_t team_params;
  ucc_context_config_h ctx_config;
  ucc_status_t status;
  ucc_lib_h lib;
  ucc_lib_config_h lib_config;

  // init ucc lib
  ucc_lib_params_t lib_params = {.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE,
      .thread_mode = UCC_THREAD_SINGLE,
      .coll_types = {},
      .reduction_types = {},
      .sync_type = {}};

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_lib_config_read(nullptr, nullptr, &lib_config));
  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_init(&lib_params, lib_config, &lib));
  ucc_lib_config_release(lib_config);

  // init ucc context
  ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB;
  // ctx_params.oob.allgather = oob_allgather;
  ctx_params.oob.allgather = redis_allgather;
  // ctx_params.oob.req_test = oob_allgather_test;
  ctx_params.oob.req_test = redis_allgather_test;

  ctx_params.oob.req_free = oob_allgather_free;
  ctx_params.oob.coll_info = (void *) &coll_info;
  ctx_params.oob.n_oob_eps = static_cast<uint32_t>(comm.GetWorldSize());
  ctx_params.oob.oob_ep = static_cast<uint32_t>(comm.GetRank());

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_context_config_read(lib, nullptr, &ctx_config));

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_context_create(lib, &ctx_params, ctx_config,
                                                       &comm.uccContext));
  ucc_context_config_release(ctx_config);

  // init ucc team
  team_params.mask = UCC_TEAM_PARAM_FIELD_OOB;
  // team_params.oob.allgather = oob_allgather;
  team_params.oob.allgather = redis_allgather;
  // team_params.oob.req_test = oob_allgather_test;
  team_params.oob.req_test = redis_allgather_test;

  team_params.oob.req_free = oob_allgather_free;
  team_params.oob.coll_info = (void *) &coll_info;
  team_params.oob.n_oob_eps = static_cast<uint32_t>(comm.GetWorldSize());
  team_params.oob.oob_ep = static_cast<uint32_t>(comm.GetRank());
  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_team_create_post(&comm.uccContext, 1, &team_params,
                                                         &comm.uccTeam));
  while (UCC_INPROGRESS == (status = ucc_team_create_test(comm.uccTeam))) {
//    RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_context_progress(comm.uccContext));
  }

  RETURN_CYLON_STATUS_IF_UCC_FAILED(status);
  return Status::OK();
}

CommType UCXUCCCommunicator::GetCommType() const {
  return UCX;
}

std::unique_ptr<Channel> UCXUCCCommunicator::CreateChannel() const {
  return ucx_comm_->CreateChannel();
}

void UCXUCCCommunicator::Finalize() {
  if (!this->IsFinalized()) {
    ucc_status_t status;
    while (uccTeam && (UCC_INPROGRESS == (status = ucc_team_destroy(uccTeam)))) {
      if (UCC_OK != status) {
        LOG(ERROR) << "ucc_team_destroy failed";
        break;
      }
    }
    ucc_context_destroy(uccContext);
    mpi_check_and_finalize();
    ucx_comm_->Finalize();
    finalized = true;
  }
}

void UCXUCCCommunicator::Barrier() {
  return ucx_comm_->Barrier();
}

Status UCXUCCCommunicator::AllGather(const std::shared_ptr<Table> &table,
                                     std::vector<std::shared_ptr<Table>> *out) const {
  ucc::UccTableAllgatherImpl impl(uccTeam, uccContext, world_size);
  return impl.Execute(table, out);
}

Status UCXUCCCommunicator::Gather(const std::shared_ptr<Table> &table,
                                  int gather_root,
                                  bool gather_from_root,
                                  std::vector<std::shared_ptr<Table>> *out) const {
  ucc::UccTableGatherImpl impl(uccTeam, uccContext, rank, world_size);
  return impl.Execute(table, gather_root, gather_from_root, out);
}

Status UCXUCCCommunicator::Bcast(std::shared_ptr<Table> *table,
                                 int bcast_root,
                                 const std::shared_ptr<CylonContext> &ctx) const {
  ucc::UccTableBcastImpl impl(uccTeam, uccContext);
  // The ctx_ptr and the real context are not the same
  return impl.Execute(table, bcast_root, ctx);
}

Status UCXUCCCommunicator::AllReduce(const std::shared_ptr<Column> &column,
                                     net::ReduceOp reduce_op,
                                     std::shared_ptr<Column> *output) const {
  ucc::UccAllReduceImpl impl(uccTeam, uccContext);
  return impl.Execute(column, reduce_op, output);
}

Status UCXUCCCommunicator::AllReduce(const std::shared_ptr<Scalar> &values,
                                     net::ReduceOp reduce_op,
                                     std::shared_ptr<Scalar> *output) const {
  ucc::UccAllReduceImpl impl(uccTeam, uccContext);
  return impl.Execute(values, reduce_op, output);
}

Status UCXUCCCommunicator::Allgather(const std::shared_ptr<Column> &values,
                                     std::vector<std::shared_ptr<Column>> *output) const {
  ucc::UccAllGatherImpl impl(uccTeam, uccContext, world_size);
  return impl.Execute(values, world_size, output);
}

Status UCXUCCCommunicator::Allgather(const std::shared_ptr<Scalar> &value,
                                     std::shared_ptr<Column> *output) const {
  ucc::UccAllGatherImpl impl(uccTeam, uccContext, world_size);
  return impl.Execute(value, world_size, output);
}

#endif // BUILD_CYLON_UCC
}  // namespace net
}  // namespace cylon

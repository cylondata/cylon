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

#include "cylon/net/ucx/ucx_communicator.hpp"

#include <glog/logging.h>

#include <memory>

#include "cylon/net/communicator.hpp"
#include "cylon/net/ucx/ucx_channel.hpp"
#include "cylon/util/macros.hpp"

#ifdef BUILD_CYLON_UCC
#include "cylon/net/ucc/ucc_operations.hpp"
#endif

#ifdef BUILD_CYLON_REDIS
#include "cylon/net/ucx/redis_ucx_ucc_oob_context.hpp"
#endif

namespace cylon {
namespace net {

static constexpr int kBarrierFlag = UINT32_MAX;

void mpi_check_and_finalize() {
  int mpi_finalized;
  MPI_Finalized(&mpi_finalized);
  if (!mpi_finalized) {
    MPI_Finalize();
  }
}

CommType UCXConfig::Type() { return CommType::UCX; }

UCXConfig::UCXConfig(std::shared_ptr<UCXOOBContext> oobContext) {
  this->oobContext = oobContext;
}

std::shared_ptr<UCXConfig> UCXConfig::Make(
    std::shared_ptr<UCXOOBContext> oobContext) {
  return std::make_shared<UCXConfig>(oobContext);
}

std::shared_ptr<UCXConfig> UCXConfig::Make(MPI_Comm comm) {
    return std::make_shared<UCXConfig>(comm);
}

UCXConfig::UCXConfig(MPI_Comm comm) : comm_(comm) {}

MPI_Comm UCXConfig::GetMPIComm() const { return comm_; }

void UCXConfig::setOOBContext(std::shared_ptr<UCXOOBContext> oobContext) {
  this->oobContext = oobContext;
}

std::shared_ptr<UCXOOBContext> UCXConfig::getOOBContext() {
  return this->oobContext;
}

#ifdef BUILD_CYLON_UCC
CommType UCCConfig::Type() { return CommType::UCC; }

UCCConfig::UCCConfig(std::shared_ptr<UCCOOBContext> oob_context)
    : oobContext(oob_context) {}

std::shared_ptr<UCCConfig> UCCConfig::Make(
    std::shared_ptr<UCCOOBContext> &oob_context) {
  return std::make_shared<UCCConfig>(oob_context);
}

void UCCConfig::setOOBContext(std::shared_ptr<UCCOOBContext> oobContext) {
  this->oobContext = oobContext;
}

std::shared_ptr<UCCOOBContext> UCCConfig::getOOBContext() { return oobContext; }


#endif

std::unique_ptr<Channel> UCXCommunicator::CreateChannel() const {
    return std::make_unique<UCXChannel>(this);
}
int UCXCommunicator::GetRank() const { return this->rank; }
int UCXCommunicator::GetWorldSize() const { return this->world_size; }

Status UCXCommunicator::AllGather(
    const std::shared_ptr<Table> &table,
    std::vector<std::shared_ptr<Table>> *out) const {
  CYLON_UNUSED(table);
  CYLON_UNUSED(out);
  return {Code::NotImplemented, "All gather not implemented for ucx"};
}

Status UCXCommunicator::Gather(const std::shared_ptr<Table> &table,
                               int gather_root, bool gather_from_root,
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

UCXCommunicator::UCXCommunicator(MemoryPool *pool)
    : Communicator(pool, -1, -1) {}

UCXCommunicator::UCXCommunicator(MemoryPool *pool, std::string address, int port)
     : UCXCommunicator(pool) {
    if (port != -1) {
        this->port = port;
        this->address = address;
        this->addressTyp = UCX_ADDRESS_TYP::SOCK;
    }
}

UCXCommunicator::UCXCommunicator(MemoryPool *pool, bool externally_init, MPI_Comm comm)
                : Communicator(pool, -1, -1),
                  externally_init(externally_init), mpi_comm(comm) {}

Status UCXCommunicator::AllReduce(const std::shared_ptr<Scalar> &values,
                                  net::ReduceOp reduce_op,
                                  std::shared_ptr<Scalar> *output) const {
  CYLON_UNUSED(values);
  CYLON_UNUSED(reduce_op);
  CYLON_UNUSED(output);
  return {Code::NotImplemented, "Allreduce not implemented for ucx"};
}

Status UCXCommunicator::Allgather(
    const std::shared_ptr<Column> &values,
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

Status UCXCommunicator::MakeOOB(const std::shared_ptr<CommConfig> &config, MemoryPool *pool,
                                std::shared_ptr<Communicator> *out, const std::shared_ptr<CommConfig> &parent_config) {
    const auto &ucc_config = std::static_pointer_cast<UCXConfig>(config);
    auto oob_context = ucc_config->getOOBContext();
    auto portStr = (+UCXConfigMapV::PORT)._to_string();

    auto portConf = parent_config->GetConfig(portStr);

    if (!portConf.empty()) {
        auto addressConf = parent_config->GetConfig((+UCXConfigMapV::ADDRESS)._to_string());
        auto port = std::stoi(portConf);
        *out = std::make_shared<UCXCommunicator>(pool, addressConf, port);

    } else {
        *out = std::make_shared<UCXCommunicator>(pool);
    }



    auto &comm = static_cast<UCXCommunicator &>(**out);
    comm.oobContext = oob_context;

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

    RETURN_CYLON_STATUS_IF_FAILED(oob_context->InitOOB());

    // Get the rank for checking send to self, and initializations
    RETURN_CYLON_STATUS_IF_FAILED(
            oob_context->getWorldSizeAndRank(comm.world_size, comm.rank));

    int rank = comm.rank, world_size = comm.world_size;

    // Init context
    RETURN_CYLON_STATUS_IF_FAILED(
            cylon::ucx::initContext(&comm.ucpContext, nullptr));

    // Init recv worker and get address
    ucpRecvWorkerAddr =
            cylon::ucx::initWorker(comm.ucpContext, &comm.ucpRecvWorker);
    // Init send worker
    ucpSendWorkerAddr =
            cylon::ucx::initWorker(comm.ucpContext, &comm.ucpSendWorker);

    //  Gather all worker addresses
    // All addresses buffer for allGather
    auto allAddresses =
            std::make_unique<uint8_t[]>(ucpRecvWorkerAddr->addrSize * world_size);

    RETURN_CYLON_STATUS_IF_FAILED(oob_context->OOBAllgather(
            (uint8_t *)ucpRecvWorkerAddr->addr, allAddresses.get(),
            (int)ucpRecvWorkerAddr->addrSize, (int)ucpRecvWorkerAddr->addrSize));

    // Iterate and set the sends
    comm.endPointMap.reserve(world_size);
    for (sIndx = 0; sIndx < world_size; sIndx++) {
        ucp_ep_params_t epParams;
        ucp_ep_h ep;

        // If not self, then check if the worker address has been received.
        //  If self,then assign local worker
        if (rank != sIndx) {
            address = reinterpret_cast<ucp_address_t *>(
                    allAddresses.get() + sIndx * ucpRecvWorkerAddr->addrSize);
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
                    "Error when creating the endpoint: " +
                    std::string(ucs_status_string(ucxStatus))};
        }
    }

    // Cleanup
    delete (ucpRecvWorkerAddr);
    delete (ucpSendWorkerAddr);

    return Status::OK();
}



Status UCXCommunicator::Make(const std::shared_ptr<CommConfig> &config,
                             MemoryPool *pool,
                             std::shared_ptr<Communicator> *out) {
    const auto &mpi_config = std::static_pointer_cast<UCXConfig>(config);
    auto mpi_comm = mpi_config->GetMPIComm();

    // MPI init
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Init(nullptr, nullptr));
    }

    if (mpi_comm == MPI_COMM_NULL) {
        mpi_comm = MPI_COMM_WORLD;
    }

    *out = std::make_shared<UCXCommunicator>(pool, initialized, mpi_comm);
    auto &comm = dynamic_cast<UCXCommunicator &>(**out);

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

    // Get the rank for checking send to self, and initializations
    MPI_Comm_rank(mpi_comm, &comm.rank);
    MPI_Comm_size(mpi_comm, &comm.world_size);

    int rank = comm.rank, world_size = comm.world_size;

    // Init context
    RETURN_CYLON_STATUS_IF_FAILED(cylon::ucx::initContext(&comm.ucpContext, nullptr));

    // Init recv worker and get address
    ucpRecvWorkerAddr = cylon::ucx::initWorker(comm.ucpContext, &comm.ucpRecvWorker);
    // Init send worker
    ucpSendWorkerAddr = cylon::ucx::initWorker(comm.ucpContext, &comm.ucpSendWorker);

    //  Gather all worker addresses
    // All addresses buffer for allGather
    auto allAddresses = std::make_unique<uint8_t[]>(ucpRecvWorkerAddr->addrSize * world_size);
    RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Allgather(ucpRecvWorkerAddr->addr,
                                                    (int) ucpRecvWorkerAddr->addrSize,
                                                    MPI_BYTE,
                                                    allAddresses.get(),
                                                    (int) ucpRecvWorkerAddr->addrSize,
                                                    MPI_BYTE,
                                                    mpi_comm));

    // Iterate and set the sends
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

    return Status::OK();
}

void UCXCommunicator::Finalize() {
  if (!externally_init && !IsFinalized()) {
    ucp_cleanup(ucpContext);
    mpi_check_and_finalize();
    finalized = true;
  }
}

void UCXCommunicator::Barrier() {
  MPI_Barrier(this->mpi_comm);
}

CommType UCXCommunicator::GetCommType() const {
  return UCX;
}



#ifdef BUILD_CYLON_UCC

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
  CYLON_UNUSED(req);
  return UCC_OK;
}

        UCXUCCCommunicator::UCXUCCCommunicator(const std::shared_ptr<Communicator> &ucx_comm)
                : Communicator(ucx_comm->GetMemoryPool(), ucx_comm->GetRank(), ucx_comm->GetWorldSize()),
                  ucx_comm_(std::static_pointer_cast<UCXCommunicator>(ucx_comm)) {}





UCXUCCCommunicator::UCXUCCCommunicator(
    std::shared_ptr<Communicator> ucx_comm,
    std::shared_ptr<UCCOOBContext> &oob_context)
    : Communicator(ucx_comm->GetMemoryPool(), ucx_comm->GetRank(),
                   ucx_comm->GetWorldSize()),
      ucx_comm_(std::static_pointer_cast<UCXCommunicator>(ucx_comm)),
      oobContext(oob_context) {}

Status UCXUCCCommunicator::MakeOOB(std::shared_ptr<UCCOOBContext> &ucc_oob_ctx,
                                   MemoryPool *pool, std::shared_ptr<Communicator> *out,
                                   const std::shared_ptr<CommConfig> &config) {
    std::shared_ptr<Communicator> ucx_comm;
    auto ucx_config =
            std::make_shared<UCXConfig>(ucc_oob_ctx->makeUCXOOBContext());

    UCXCommunicator::MakeOOB(ucx_config, pool, &ucx_comm, config);



    *out = std::make_shared<UCXUCCCommunicator>(std::move(ucx_comm), ucc_oob_ctx);

    auto &comm = *std::static_pointer_cast<UCXUCCCommunicator>(*out);
    comm.oobContext = ucc_oob_ctx;
    comm.oobContext->InitOOB(comm.GetRank());

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

    RETURN_CYLON_STATUS_IF_UCC_FAILED(
            ucc_lib_config_read(nullptr, nullptr, &lib_config));
    RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_init(&lib_params, lib_config, &lib));
    ucc_lib_config_release(lib_config);

    // init ucc context
    ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB;

    if (ucc_oob_ctx->Type() == OOBType::OOB_REDIS) {
#ifdef BUILD_CYLON_REDIS
        ctx_params.oob.allgather = team_params.oob.allgather =
        UCCRedisOOBContext::oob_allgather;
    ctx_params.oob.req_test = team_params.oob.req_test =
        UCCRedisOOBContext::oob_allgather_test;
    ctx_params.oob.req_free = team_params.oob.req_free =
        UCCRedisOOBContext::oob_allgather_free;
#else
        ctx_params.oob.allgather = team_params.oob.allgather =
                UCCMPIOOBContext::oob_allgather;
        ctx_params.oob.req_test = team_params.oob.req_test =
                UCCMPIOOBContext::oob_allgather_test;
        ctx_params.oob.req_free = team_params.oob.req_free =
                UCCMPIOOBContext::oob_allgather_free;
#endif
    } else if (ucc_oob_ctx->Type() == OOBType::OOB_MPI) {
        ctx_params.oob.allgather = team_params.oob.allgather =
                UCCMPIOOBContext::oob_allgather;
        ctx_params.oob.req_test = team_params.oob.req_test =
                UCCMPIOOBContext::oob_allgather_test;
        ctx_params.oob.req_free = team_params.oob.req_free =
                UCCMPIOOBContext::oob_allgather_free;
    } else {
        return {Code::NotImplemented, "UCC OOB communication type not supported."};
    }

    ctx_params.oob.coll_info = ucc_oob_ctx->getCollInfo();

    ctx_params.oob.n_oob_eps = static_cast<uint32_t>(comm.GetWorldSize());
    ctx_params.oob.oob_ep = static_cast<uint32_t>(comm.GetRank());

    RETURN_CYLON_STATUS_IF_UCC_FAILED(
            ucc_context_config_read(lib, nullptr, &ctx_config));

    RETURN_CYLON_STATUS_IF_UCC_FAILED(
            ucc_context_create(lib, &ctx_params, ctx_config, &comm.uccContext));

    ucc_context_config_release(ctx_config);

    // init ucc team
    team_params.mask = UCC_TEAM_PARAM_FIELD_OOB;

    team_params.oob.coll_info = ucc_oob_ctx->getCollInfo();

    team_params.oob.n_oob_eps = static_cast<uint32_t>(comm.GetWorldSize());
    team_params.oob.oob_ep = static_cast<uint32_t>(comm.GetRank());
    RETURN_CYLON_STATUS_IF_UCC_FAILED(
            ucc_team_create_post(&comm.uccContext, 1, &team_params, &comm.uccTeam));

    while (UCC_INPROGRESS == (status = ucc_team_create_test(comm.uccTeam))) {
        //    RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_context_progress(comm.uccContext));
    }

    RETURN_CYLON_STATUS_IF_UCC_FAILED(status);
    return Status::OK();

}


Status UCXUCCCommunicator::Make(const std::shared_ptr<CommConfig> &config,
                                MemoryPool *pool,
                                std::shared_ptr<Communicator> *out) {
  auto ucc_config = std::static_pointer_cast<UCCConfig>(config);
  auto ucc_oob_ctx = ucc_config->getOOBContext();

  if (ucc_oob_ctx != nullptr) { //call MPI Com
      return UCXUCCCommunicator::MakeOOB(ucc_oob_ctx, pool, out, config);
  }

          std::shared_ptr<Communicator> ucx_comm;
          RETURN_CYLON_STATUS_IF_FAILED(UCXCommunicator::Make(config, pool, &ucx_comm));

          *out = std::make_shared<UCXUCCCommunicator>(std::move(ucx_comm));
          auto &comm = *std::static_pointer_cast<UCXUCCCommunicator>(*out);

          auto mpi_comm = comm.ucx_comm_->mpi_comm;

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
          ctx_params.oob.allgather = oob_allgather;
          ctx_params.oob.req_test = oob_allgather_test;
          ctx_params.oob.req_free = oob_allgather_free;
          ctx_params.oob.coll_info = (void *) mpi_comm;
          ctx_params.oob.n_oob_eps = static_cast<uint32_t>(comm.GetWorldSize());
          ctx_params.oob.oob_ep = static_cast<uint32_t>(comm.GetRank());

          RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_context_config_read(lib, nullptr, &ctx_config));

          RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_context_create(lib, &ctx_params, ctx_config,
                                                               &comm.uccContext));
          ucc_context_config_release(ctx_config);

          // init ucc team
          team_params.mask = UCC_TEAM_PARAM_FIELD_OOB;
          team_params.oob.allgather = oob_allgather;
          team_params.oob.req_test = oob_allgather_test;
          team_params.oob.req_free = oob_allgather_free;
          team_params.oob.coll_info = (void *) mpi_comm;
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

CommType UCXUCCCommunicator::GetCommType() const { return UCX; }

std::unique_ptr<Channel> UCXUCCCommunicator::CreateChannel() const {
  return ucx_comm_->CreateChannel();
}

void UCXUCCCommunicator::Finalize() {
  if (!this->IsFinalized()) {
    ucc_status_t status;
    while (uccTeam &&
           (UCC_INPROGRESS == (status = ucc_team_destroy(uccTeam)))) {
      if (UCC_OK != status) {
        LOG(ERROR) << "ucc_team_destroy failed";
        break;
      }
    }

    if (!ucx_comm_->externally_init){
      ucc_context_destroy(uccContext);
    }

    ucx_comm_->Finalize(); // this will handle MPI_Finalize
    finalized = true;
  }
}

void UCXUCCCommunicator::Barrier() {
  ucc_coll_args_t args_;
  ucc_coll_req_h req;

  args_.mask = 0;
  args_.coll_type = UCC_COLL_TYPE_BARRIER;

  ucc_status_t status;
  status = ucc_collective_init(&args_, &req, uccTeam);
  assert(status == UCC_OK);

  status = ucc_collective_post(req);
  assert(status == UCC_OK);

  status = ucc_collective_test(req);
  while (status > UCC_OK) { // loop until status == 0
    status = ucc_context_progress(uccContext);
    assert(status >= UCC_OK); // should be 0, 1 or 2

    status = ucc_collective_test(req);
  }
  assert(status == UCC_OK);

  status = ucc_collective_finalize(req);
  assert(status == UCC_OK);
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

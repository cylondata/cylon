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

#ifndef CYLON_CPP_SRC_CYLON_NET_UCX_UCX_OPERATIONS_HPP_
#define CYLON_CPP_SRC_CYLON_NET_UCX_UCX_OPERATIONS_HPP_

#include <mpi.h>
#include <ucp/api/ucp.h>

#include "cylon/status.hpp"

namespace cylon {
namespace ucx {
/**
 * Hold the completion status of a communication
 * completed - if completed show 1, else 0
 */
struct ucxContext {
  int completed;
};

/**
 * Hold the data related to a communication endpoint
 * addr - the address of the ucp worker
 * addrSize - the size of the worker address
 */
struct ucxWorkerAddr {
  ucp_address_t *addr;
  size_t addrSize;
};

/**
 * Create a UCP worker on the given UCP context.
 * @param [in] ucpContext - The context to be passed to init the worker
 * @param [out] ucpWorker - The UCP worker
 */
ucxWorkerAddr* initWorker(ucp_context_h ucpContext,
                      ucp_worker_h *ucpWorker);

/**
 * Initialize a default UCP context.
 * @param [out] ucpContext - The UCP context
 * @param [out] config - The configuration descriptor
 */
Status initContext(ucp_context_h *ucpContext,
                ucp_config_t *config);
}
}
#endif //CYLON_CPP_SRC_CYLON_NET_UCX_UCX_OPERATIONS_HPP_

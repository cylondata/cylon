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

#include <iostream>
#include <string>
#include <cstring>
#include <glog/logging.h>

#include "cylon/net/ucx/ucx_operations.hpp"

/**
 * Create a UCP worker on the given UCP context.
 * @param [in] ucpContext - The context to be passed to init the worker
 * @param [out] ucpWorker - The UCP worker
 */
cylon::ucx::ucxWorkerAddr *cylon::ucx::initWorker(ucp_context_h ucpContext,
                                                  ucp_worker_h *ucpWorker) {
  // UCP objects
  // Worker params - Tuning parameters for the UCP worker.
  // Has threading and CPU count etc
  ucp_worker_params_t workerParams;
  // Variable to check status
  ucs_status_t status;

  // New worker
  auto worker = new ucxWorkerAddr();

  // Init values to worker params
  memset(&workerParams, 0, sizeof(workerParams));

  // Thread mode params
  workerParams.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  workerParams.thread_mode = UCS_THREAD_MODE_SINGLE;

  // Create UCP worker - 1:many -> context:worker
  // In - Context, worker params
  // Out - UCP worker
  status = ucp_worker_create(ucpContext, &workerParams, ucpWorker);

  // Check status of worker
  if (status != UCS_OK) {
    LOG(FATAL) << "Failed to create a UCP worker for the given UCP context: " << ucs_status_string(status);
    goto err_cleanup;
  }

  status = ucp_worker_get_address(*ucpWorker,
                                  &(worker->addr),
                                  &(worker->addrSize));
  // Check status of worker
  if (status != UCS_OK) {
    LOG(FATAL) << "Failed to get the address of the given UCP worker: " << ucs_status_string(status);
    goto err_worker;
  }

  return worker;

  err_cleanup:
  ucp_cleanup(ucpContext);
  goto final;
  err_worker:
  ucp_worker_destroy(*ucpWorker);
  goto final;
  final:
  return nullptr;
}

/**
 * Initialize a default UCP context.
 * @param [out] ucpContext - The UCP context
 * @param [out] config - The configuration descriptor
 * @return int - Status of the context init
 */
cylon::Status cylon::ucx::initContext(ucp_context_h *ucpContext,
                                      ucp_config_t *config) {
  // UCP params - The structure defines the parameters that are used for
  // UCP library tuning during UCP library "initialization".
  ucp_params_t ucpParams;
  // UCP status - To hold the status of the UCP objects
  ucs_status_t status;

  // Init UCP Params
  std::memset(&ucpParams, 0, sizeof(ucpParams));

  // The enumeration allows specifying which fields in ucp_params_t are
  // present.
  ucpParams.field_mask   = UCP_PARAM_FIELD_FEATURES |
                            UCP_PARAM_FIELD_REQUEST_SIZE |
                            UCP_PARAM_FIELD_REQUEST_INIT;
  // Set support for tags
  ucpParams.features     = UCP_FEATURE_TAG;
  ucpParams.request_size = sizeof(struct ucxContext);


  // Init UCP context
  // Inp - params, config
  // Out - UCP context
  status = ucp_init(&ucpParams, config, ucpContext);

  // Check context init
  if (status != UCS_OK) {
    return {Code::ExecutionError,
            "Failed to initialize UCP context: " + std::string(ucs_status_string(status))};
  }

  return Status::OK();
}




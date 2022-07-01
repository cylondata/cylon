/*
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/**
 * temporary example to test an issue; removed after issue resolved
 * run the example as follows:
 *  mpirun -n 4 bin/ucc_allgather_vector_example
 */

#include <mpi.h>
#include <ucc/api/ucc.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#define STR(x) #x
#define UCC_CHECK(_call)                                    \
  if (UCC_OK != (_call)) {                                  \
    fprintf(stderr, "*** UCC TEST FAIL: %s\n", STR(_call)); \
    MPI_Abort(MPI_COMM_WORLD, -1);                          \
  }

static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                  void *coll_info, void **req) {
  auto comm = (MPI_Comm)coll_info;
  MPI_Request request;

  MPI_Iallgather(sbuf, (int)msglen, MPI_BYTE, rbuf, (int)msglen, MPI_BYTE, comm,
                 &request);
  *req = (void *)request;
  return UCC_OK;
}

static ucc_status_t oob_allgather_test(void *req) {
  auto request = (MPI_Request)req;
  int completed;

  MPI_Test(&request, &completed, MPI_STATUS_IGNORE);
  return completed ? UCC_OK : UCC_INPROGRESS;
}

static ucc_status_t oob_allgather_free(void *req) { return UCC_OK; }

/* Creates UCC team for a group of processes represented by MPI
   communicator. UCC API provides different ways to create a team,
   one of them is to use out-of-band (OOB) allgather provided by
   the calling runtime. */
static ucc_team_h create_ucc_team(MPI_Comm comm, ucc_context_h ctx) {
  int rank, size;
  ucc_team_h team;
  ucc_team_params_t team_params;
  ucc_status_t status;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  team_params.mask = UCC_TEAM_PARAM_FIELD_OOB | UCC_TEAM_PARAM_FIELD_EP |
                     UCC_TEAM_PARAM_FIELD_EP_RANGE;
  team_params.oob.allgather = oob_allgather;
  team_params.oob.req_test = oob_allgather_test;
  team_params.oob.req_free = oob_allgather_free;
  team_params.oob.coll_info = (void *)comm;
  team_params.oob.n_oob_eps = size;
  team_params.oob.oob_ep = rank;
  team_params.ep = rank;
  team_params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;

  UCC_CHECK(ucc_team_create_post(&ctx, 1, &team_params, &team));
  while (UCC_INPROGRESS == (status = ucc_team_create_test(team))) {
    UCC_CHECK(ucc_context_progress(ctx));
  };
  if (UCC_OK != status) {
    fprintf(stderr, "failed to create ucc team\n");
    MPI_Abort(MPI_COMM_WORLD, status);
  }
  return team;
}

int main(int argc, char **argv) {
  ucc_lib_config_h lib_config;
  ucc_context_config_h ctx_config;
  int rank, size, i;
  ucc_team_h team;
  ucc_context_h ctx;
  ucc_lib_h lib;
  size_t msglen;
  ucc_coll_req_h req;
  ucc_coll_args_t args;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* Init ucc library */
  ucc_lib_params_t lib_params = {.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE,
                                 .thread_mode = UCC_THREAD_SINGLE,
                                 .coll_types = {},
                                 .reduction_types = {},
                                 .sync_type = {}};
  UCC_CHECK(ucc_lib_config_read(nullptr, nullptr, &lib_config));
  UCC_CHECK(ucc_init(&lib_params, lib_config, &lib));
  ucc_lib_config_release(lib_config);

  /* Init ucc context for a specified UCC_TEST_TLS */
  ucc_context_params_t ctx_params;
  ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB;
  ctx_params.oob.allgather = oob_allgather;
  ctx_params.oob.req_test = oob_allgather_test;
  ctx_params.oob.req_free = oob_allgather_free;
  ctx_params.oob.coll_info = (void *)MPI_COMM_WORLD;
  ctx_params.oob.n_oob_eps = static_cast<uint32_t>(size);
  ctx_params.oob.oob_ep = static_cast<uint32_t>(rank);

  UCC_CHECK(ucc_context_config_read(lib, nullptr, &ctx_config));
  UCC_CHECK(ucc_context_create(lib, &ctx_params, ctx_config, &ctx));

  while (UCC_OK != ucc_context_progress(ctx)) {
  }
  ucc_context_config_release(ctx_config);

  team = create_ucc_team(MPI_COMM_WORLD, ctx);

  std::vector<uint64_t> count(size), disp(size);
  std::vector<int> send(10), receive(size * 10);

  for(int i = 0; i < 10; i++) {
    send[i] = i + rank * 10 + 1;
  }

  for(int i = 0; i < size; i++) {
    count[i] = 10;
    disp[i] = 10 * i;
  }

  args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
  args.flags =
      UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT | UCC_COLL_ARGS_FLAG_COUNT_64BIT;
  args.coll_type = UCC_COLL_TYPE_ALLGATHERV;
  args.src.info.buffer = send.data();
  args.src.info.count = 10;
  args.src.info.datatype = UCC_DT_INT32;
  args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

  args.dst.info_v.buffer = receive.data();
  args.dst.info_v.counts = count.data();
  args.dst.info_v.displacements = disp.data();
  args.dst.info_v.datatype = UCC_DT_INT32;
  args.dst.info_v.mem_type = UCC_MEMORY_TYPE_HOST;

  UCC_CHECK(ucc_collective_init(&args, &req, team));
  UCC_CHECK(ucc_collective_post(req));
  while (UCC_INPROGRESS == ucc_collective_test(req)) {
    UCC_CHECK(ucc_context_progress(ctx));
  }
  ucc_collective_finalize(req);

  /* Check result */
  int sum = ((size + 1) * size) / 2;
  for (i = 0; i < 10 * size; i++) {
    std::cout<<receive[i]<<" ";
  }

  std::cout<<std::endl;

  /* Cleanup UCC */
  UCC_CHECK(ucc_team_destroy(team));
  UCC_CHECK(ucc_context_destroy(ctx));
  UCC_CHECK(ucc_finalize(lib));

  MPI_Finalize();

  return 0;
}

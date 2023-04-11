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

#ifndef CYLON_SRC_CYLON_COMM_UCXUCCOOBCONTEXTS_H_
#define CYLON_SRC_CYLON_COMM_UCXUCCOOBCONTEXTS_H_
#include <cylon/net/comm_config.hpp>
#include <cylon/net/communicator.hpp>
#include <cylon/net/ucx/ucx_operations.hpp>
#include <cylon/net/ucx/oob_type.hpp>

#include "cylon/util/macros.hpp"

#include <cylon/net/ucx/ucx_ucc_oob_context.hpp>

#ifdef BUILD_CYLON_UCC
#include <ucc/api/ucc.h>
#endif

namespace cylon {
namespace net {




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
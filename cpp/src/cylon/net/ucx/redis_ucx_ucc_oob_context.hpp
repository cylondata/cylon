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

#ifndef CYLON_REDIS_UCX_UCC_OOB_CONTEXT_HPP
#define CYLON_REDIS_UCX_UCC_OOB_CONTEXT_HPP

#include <cylon/net/comm_config.hpp>
#include <cylon/net/communicator.hpp>
#include <cylon/net/ucx/ucx_operations.hpp>
#include <cylon/net/ucx/oob_type.hpp>

#include "cylon/util/macros.hpp"

#include <cylon/net/ucx/ucx_ucc_oob_context.hpp>
#include <cylon/net/ucx/mpi_ucx_ucc_oob_context.hpp>

#ifdef BUILD_CYLON_REDIS

#include "sw/redis++/redis++.h"

#endif

#ifdef BUILD_CYLON_UCC

#include <ucc/api/ucc.h>

#endif

namespace cylon {
    namespace net {
#ifdef BUILD_CYLON_REDIS

        class UCXRedisOOBContext : public UCXOOBContext {
        public:
            UCXRedisOOBContext(int world_size, std::string redis_addr);

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

        class UCCRedisOOBContext : public UCCOOBContext {
        public:
            void InitOOB(int rank) override;

            std::shared_ptr<UCXOOBContext> makeUCXOOBContext() override;

            void *getCollInfo() override;

            OOBType Type() override;

            UCCRedisOOBContext(int world_size, std::string redis_addr);

            /***
             * This constructor is used with python script `run_ucc_with_redis.py`
             * Extracts environment variables set by the script and initializes metadata
             */
            UCCRedisOOBContext();

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
            std::string redis_addr;
        };

#endif


    }
}


#endif //CYLON_REDIS_UCX_UCC_OOB_CONTEXT_HPP


#include "redis_ucx_ucc_oob_context.hpp"

namespace cylon {
    namespace net {
#ifdef BUILD_CYLON_REDIS
        UCXRedisOOBContext::UCXRedisOOBContext(int ws, std::string rds)
                : redis(std::make_shared<sw::redis::Redis>(rds)), world_size(ws) {}

        Status UCXRedisOOBContext::InitOOB() { return Status::OK(); };

        Status UCXRedisOOBContext::getWorldSizeAndRank(int &world_size, int &rank) {
            world_size = this->world_size;
            int num_cur_processes = redis->incr("num_cur_processes");
            rank = this->rank = num_cur_processes - 1;

            return Status::OK();
        }

        Status UCXRedisOOBContext::OOBAllgather(uint8_t *src, uint8_t *dst,
                                                size_t srcSize, size_t dstSize) {
            CYLON_UNUSED(dstSize);
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

        Status UCXRedisOOBContext::Finalize() { return Status::OK(); }

        std::shared_ptr<UCXRedisOOBContext> UCXRedisOOBContext::Make(int world_size, std::string redis_addr) {
            return std::make_shared<UCXRedisOOBContext>(world_size, redis_addr);
        };

        void UCCRedisOOBContext::InitOOB(int rank) { this->rank = rank; }

        std::shared_ptr<UCXOOBContext> UCCRedisOOBContext::makeUCXOOBContext() {
            return std::make_shared<UCXRedisOOBContext>(world_size, redis_addr);
        }

        void *UCCRedisOOBContext::getCollInfo() { return this; }

        ucc_status_t UCCRedisOOBContext::oob_allgather(void *sbuf, void *rbuf,
                                                       size_t msglen, void *coll_info,
                                                       void **req) {
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
                    memcpy((uint8_t*)rbuf + i * msglen, s.data(), msglen);
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

                    memcpy((uint8_t*)rbuf + i * msglen, val.value().data(), msglen);
                }
            }

            return UCC_OK;
        }

        UCCRedisOOBContext::UCCRedisOOBContext(int ws,
                                               std::string rds)
                : world_size(ws), redis(std::make_shared<sw::redis::Redis>(rds)), redis_addr(rds) {}

        UCCRedisOOBContext::UCCRedisOOBContext() {
            redis_addr = "tcp://" + std::string(getenv("CYLON_UCX_OOB_REDIS_ADDR"));
            world_size = std::atoi(getenv("CYLON_UCX_OOB_WORLD_SIZE"));
            redis = std::make_shared<sw::redis::Redis>(redis_addr);
        }

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

        std::shared_ptr<UCCRedisOOBContext> UCCRedisOOBContext::Make(int world_size, std::string redis_addr) {
            return std::make_shared<UCCRedisOOBContext>(world_size, redis_addr);
        }

        Status UCCRedisOOBContext::Finalize() {

            return Status::OK();

        }

        void UCCRedisOOBContext::clearDB() {
            if (this->redis != nullptr) {
                this->redis->flushdb(false);
            }
        }

#endif

    }
}
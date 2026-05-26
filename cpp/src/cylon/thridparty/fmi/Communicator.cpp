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

#include <sw/redis++/connection.h>
#include <sw/redis++/redis.h>
#include "Communicator.hpp"
#include <glog/logging.h>

FMI::Communicator::Communicator(const FMI::Utils::peer_num peer_id, const FMI::Utils::peer_num num_peers,
                                const std::shared_ptr<FMI::Utils::Backends> &backend, const std::string comm_name,
                                std::string redis_host, int redis_port, const std::string redis_namespace,
                                int ttl_seconds, int s3_retry_initial_ms, int s3_retry_max_ms) {

    this->peer_id = peer_id;
    this->num_peers = num_peers;
    this->s3_retry_initial_ms_ = s3_retry_initial_ms;
    this->s3_retry_max_ms_ = s3_retry_max_ms;
    if (!redis_namespace.empty()) {
        // Use "/" separator for S3 to create folder structure, ":" for Redis/Direct
        auto backend_type = backend->getBackendType();
        std::string sep = (backend_type == FMI::Utils::BackendType::S3) ? "/" : ":";
        this->comm_name = redis_namespace + sep + comm_name;
    } else {
        this->comm_name = comm_name;
    }


    if (redis_port > 0 && !redis_host.empty()) {
        //override rank with redis to dynamically determine similar to ucc/ucx integration
        auto opts = sw::redis::ConnectionOptions{};
        opts.host = redis_host;
        opts.port = redis_port;
        auto redis = std::make_shared<sw::redis::Redis>(opts);

        std::string key;
        if (!redis_namespace.empty()) {
            key = std::string(redis_namespace + "_" + "num_cur_processes");
        } else {
            key = "num_cur_processes";
        }
        int num_cur_processes = redis->incr(key);
        redis->expire(key, std::chrono::seconds(ttl_seconds));
        this->peer_id = num_cur_processes - 1;
        LOG(INFO) << "current rank from redis: " <<  this->peer_id;

    }

    auto backend_name = backend->getName();
    channel = Comm::Channel::get_channel(backend);
    channel->set_redis_host(redis_host);
    channel->set_redis_port(redis_port);

    register_channel(backend_name, channel, Utils::DEFAULT);
    channel->set_key_ttl(ttl_seconds);
    channel->set_s3_retry_initial_ms(s3_retry_initial_ms_);
    channel->set_s3_retry_max_ms(s3_retry_max_ms_);
    channel->init();
}

void FMI::Communicator::register_channel(std::string name, std::shared_ptr<FMI::Comm::Channel> c,
                                         Utils::Operation op) {
    c->set_peer_id(peer_id);
    c->set_num_peers(num_peers);
    c->set_comm_name(comm_name);

}

FMI::Communicator::~Communicator() {
    channel->finalize();
}

FMI::Utils::peer_num FMI::Communicator::getNumPeers() const {
    return num_peers;
}

FMI::Utils::peer_num FMI::Communicator::getPeerId() const {
    return peer_id;
}

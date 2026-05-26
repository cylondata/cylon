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

#include <cstring>
#include "cylon/thridparty/fmi/comm/Channel.hpp"
#include "S3.hpp"
#include "Redis.hpp"
#include "Direct.hpp"


std::shared_ptr<FMI::Comm::Channel>
FMI::Comm::Channel::get_channel(const std::shared_ptr<FMI::Utils::Backends> &backend) {
    auto my_backend = backend.get();

    if (my_backend->getBackendType() == FMI::Utils::BackendType::S3) {
        return std::make_shared<S3>(backend);
    }
#ifdef BUILD_CYLON_REDIS
    else if (my_backend->getBackendType() == FMI::Utils::BackendType::Redis) {
        return std::make_shared<Redis>(backend);
    }
#endif
    else if (my_backend->getBackendType() == FMI::Utils::BackendType::Direct) {
        return std::make_shared<Direct>(backend);
    }
    else {
        throw std::runtime_error("Unknown channel name passed");
    }
}
void FMI::Comm::Channel::gather(const std::shared_ptr<channel_data> sendbuf,
                                std::shared_ptr<channel_data> recvbuf, FMI::Utils::peer_num root) {
    if (peer_id != root) {
        send(sendbuf, root);
    } else {
        auto buffer_length = sendbuf->len;
        for (int i = 0; i < num_peers; i++) {
            if (i == root) {
                std::memcpy(recvbuf->buf.get() + root * buffer_length, sendbuf->buf.get(), buffer_length);
            } else {
                auto peer_data = std::make_shared<channel_data>(recvbuf->buf.get()
                        + i * buffer_length, buffer_length);
                recv(peer_data, i);
            }
        }
    }
}




void FMI::Comm::Channel::scatter(const std::shared_ptr<channel_data> sendbuf,
                                 std::shared_ptr<channel_data> recvbuf, FMI::Utils::peer_num root) {
    if (peer_id == root) {
        auto buffer_length = recvbuf->len;
        for (int i = 0; i < num_peers; i++) {
            if (i == root) {
                std::memcpy(recvbuf->buf.get(), sendbuf->buf.get() + root * buffer_length,
                            buffer_length);
            } else {
                auto peer_data = std::make_shared<channel_data>(sendbuf->buf.get()
                        + i * buffer_length, buffer_length);
                send(peer_data, i);
            }
        }
    } else {
        recv(recvbuf, root);
    }
}

void FMI::Comm::Channel::allreduce(const std::shared_ptr<channel_data> sendbuf,
                                   std::shared_ptr<channel_data> recvbuf, raw_function f) {
    reduce(sendbuf, recvbuf, 0, f);
    bcast(recvbuf, 0);
}

void FMI::Comm::Channel::allgather(const std::shared_ptr<channel_data> sendbuf,
                                   std::shared_ptr<channel_data> recvbuf,
                                   FMI::Utils::peer_num root) {
    allgather(sendbuf, recvbuf, root, Utils::BLOCKING, nullptr);
}

void FMI::Comm::Channel::allgather(const std::shared_ptr<channel_data> sendbuf,
                                   std::shared_ptr<channel_data> recvbuf,
                                   FMI::Utils::peer_num root,
                                   Utils::Mode mode,
                                   std::function<void(FMI::Utils::NbxStatus, const std::string&,
                                                      FMI::Utils::fmiContext *)> callback) {}

void FMI::Comm::Channel::allgatherv(const std::shared_ptr<channel_data> sendbuf,
                                    std::shared_ptr<channel_data> recvbuf, FMI::Utils::peer_num root,
                                    const std::vector<int32_t> &recvcounts,
                                    const std::vector<int32_t> &displs) {
    allgatherv(sendbuf, recvbuf, root, recvcounts, displs, Utils::BLOCKING, nullptr);
}

void
FMI::Comm::Channel::allgatherv(const std::shared_ptr<channel_data> sendbuf,
                               std::shared_ptr<channel_data> recvbuf, FMI::Utils::peer_num root,
                                   const std::vector<int32_t> &recvcounts, const std::vector<int32_t> &displs,
                                   Utils::Mode mode,
                                   std::function<void(FMI::Utils::NbxStatus, const std::string&,
                                                      FMI::Utils::fmiContext *)> callback) {}

void FMI::Comm::Channel::gatherv(const std::shared_ptr<channel_data> sendbuf,
                                 std::shared_ptr<channel_data> recvbuf,
                                 FMI::Utils::peer_num root,
                                 const std::vector<int32_t> &recvcounts,
                                 const std::vector<int32_t> &displs) {
    gatherv(sendbuf, recvbuf, root, recvcounts, displs, Utils::BLOCKING, nullptr);

}

void FMI::Comm::Channel::gatherv(const std::shared_ptr<channel_data> sendbuf,
                                 std::shared_ptr<channel_data> recvbuf,
                                 FMI::Utils::peer_num root,
                                 const std::vector<int32_t> &recvcounts,
                                 const std::vector<int32_t> &displs,
                                 Utils::Mode mode, std::function<void(FMI::Utils::NbxStatus, const std::string&,
                                                                      FMI::Utils::fmiContext *)> callback) {

}

void FMI::Comm::Channel::bcast(std::shared_ptr<channel_data> buf, FMI::Utils::peer_num root) {
    bcast(buf, root, Utils::BLOCKING, nullptr);

}

void FMI::Comm::Channel::bcast(std::shared_ptr<channel_data> buf, FMI::Utils::peer_num root, FMI::Utils::Mode mode,
                               std::function<void(FMI::Utils::NbxStatus, const std::string &,
                                                  FMI::Utils::fmiContext *)> callback) {

}

int FMI::Comm::Channel::getMaxTimeout() {
    return -1;
}


void FMI::Comm::Channel::init() {
//noop
}

bool FMI::Comm::Channel::checkReceive(FMI::Utils::peer_num dest, Utils::Mode mode) {
    return true;
}

bool FMI::Comm::Channel::checkSend(FMI::Utils::peer_num dest, Utils::Mode mode) {
    return true;
}

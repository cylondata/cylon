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
#include "Direct.hpp"

#include "../../TCPunch/client/tcpunch.hpp"
#include <sys/epoll.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <cstring>
#include <cerrno>
#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include "../utils/DirectBackend.hpp"
#include "cylon/net/channel.hpp"
#include <sw/redis++/redis++.h>



#include <glog/logging.h>
#include <sys/poll.h>

#define holepunch_connect_to 120000
#define max_tcpunch_tries 6



FMI::Comm::Direct::Direct(const std::shared_ptr<FMI::Utils::Backends> &backend) {
    struct addrinfo hints, *res, *p;
    int status;
    char ipstr[INET6_ADDRSTRLEN];

    auto direct_backend = dynamic_cast<FMI::Utils::DirectBackend *>(backend.get());

    hostname = direct_backend->getHost();
    port = direct_backend->getPort();
    mode = direct_backend->getBlockingMode();
    enable_ping = direct_backend->enableHostPing();
    if (direct_backend->resolveHostDNS()) {

        memset(&hints, 0, sizeof hints);
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;

        if ((status = getaddrinfo(hostname.c_str(), nullptr, &hints, &res)) != 0) {
            LOG(ERROR)  << "getaddrinfo error: " << gai_strerror(status) << std::endl;
        } else {
            // Iterate through the result list and convert each address to a string
            for(p = res; p != nullptr; p = p->ai_next) {
                void *addr;

                // Get the pointer to the address itself,
                struct sockaddr_in *ipv4 = (struct sockaddr_in *)p->ai_addr;
                addr = &(ipv4->sin_addr);

                // Convert the IP to a string and print it:
                inet_ntop(p->ai_family, addr, ipstr, sizeof ipstr);

                //TODO: remove
                std::cout << " resolved rendezvous DNS: " << ipstr << std::endl;
            }

            freeaddrinfo(res); // Free the linked list
            hostname = ipstr;

        }

    }

    max_timeout = direct_backend->getMaxTimeout();



    sockets[Utils::NONBLOCKING] = {};
    sockets[Utils::BLOCKING] = {};



    io_states[Utils::Operation::SEND] = {};
    io_states[Utils::Operation::RECEIVE] = {};


}

inline const char* ModeToString(FMI::Utils::Mode mode) {
    switch (mode) {
        case FMI::Utils::Mode::BLOCKING: return "BLOCKING";
        case FMI::Utils::Mode::NONBLOCKING: return "NONBLOCKING";

        default: return "UNKNOWN_MODE";
    }
}

void FMI::Comm::Direct::start_holepunch_subscriber() {
    std::thread([this]() {
        if (redis_port > 0 && !redis_host.empty()) {
            auto opts = sw::redis::ConnectionOptions{};
            opts.host = redis_host;
            opts.port = redis_port;
            auto redis = std::make_shared<sw::redis::Redis>(opts);
            auto sub = redis->subscriber();

            sub.on_message([this](const std::string &channel, const std::string &msg) {
                int from = -1, to = -1;
                LOG(INFO) << "received message from publisher: " << msg;
                sscanf(msg.c_str(), "from:%d,to:%d", &from, &to);
                if (to == this->peer_id) {
                    LOG(INFO) << "Received reverse connect request from peer " << from;
                    // Trigger a connect attempt from this node to the sender
                    std::string pairing = get_pairing_name(this->peer_id, from, Utils::BLOCKING);
                    try {
                        check_socket(from, pairing);  // This will do the actual reverse connect
                    } catch (const Utils::Timeout &) {
                        LOG(WARNING) << "Reverse connect to " << from << " failed.";
                    }
                }

            });


            sub.subscribe("fmi_connect");

            try {
                while (true) sub.consume();  // Blocking wait
            } catch (const std::exception &e) {
                LOG(ERROR) << "Redis subscribe error: " << e.what();
            }
        }
    }).detach();
}

void FMI::Comm::Direct::init_blocking_sockets() {
    if (num_peers> 0) {

        LOG(INFO) << "init blocking sockets";

        for (int i = 0; i < num_peers; ++i) {

            if (i == peer_id) continue;

            std::string send_pairing_b = get_pairing_name(peer_id, i, Utils::BLOCKING);

            check_socket(i, send_pairing_b);

        }

    }
    blocking_init = true;
}

void FMI::Comm::Direct::start_ping_thread(Utils::Mode mode) {

    std::thread([this, mode]() {

            for (int i = 0; i < num_peers; ++i) {
                if (i == peer_id) continue;
                if (sockets[mode][i] != -1) {
                    try {
                        PingMessage ping{};
                        ::send(sockets[mode][i], &ping, sizeof(ping), 0);
                        LOG(INFO) << "Sent PING to peer " << i << " Mode: " << ModeToString(mode);


                    } catch (...) {
                        LOG(ERROR) << "PING send failed to peer " << i << " Mode: " << ModeToString(mode);;
                    }
                }
            }
            return;

    }).detach();

}


void FMI::Comm::Direct::init() {
    //iterator over world size and create all sockets for non-blocking based on multi-send/receives
    //create all the connections
    //start_holepunch_subscriber();
    if (num_peers> 0) {

        for (int i = 0; i < num_peers; ++i) {

            if (i == peer_id) continue;


            if (mode == Utils::NONBLOCKING) {
                std::string send_pairing_nb = get_pairing_name(peer_id, i, Utils::NONBLOCKING);
                check_socket_nbx(i, send_pairing_nb);
            }

            //always create a pair of blocking sockets
            //std::string send_pairing_b = get_pairing_name(peer_id, i, Utils::BLOCKING);

            //check_socket(i, send_pairing_b);

        }


        if (mode == Utils::NONBLOCKING && enable_ping) {
            start_ping_thread(Utils::NONBLOCKING);
        }
    }
}

FMI::Comm::Direct::~Direct() {
    for (auto sock : sockets[Utils::BLOCKING]) if (sock != -1) close(sock);
    for (auto sock : sockets[Utils::NONBLOCKING]) if (sock != -1) close(sock);


}

std::string FMI::Comm::Direct::get_pairing_name(FMI::Utils::peer_num a,
                                                FMI::Utils::peer_num b,
                                                FMI::Utils::Mode mode) {
    int min_id = std::min(a, b);
    int max_id = std::max(a, b);
    return "fmi_pair" + std::to_string(min_id) + "_" + std::to_string(max_id) + ModeToString(mode);
}

void FMI::Comm::Direct::send_object_blocking2(std::shared_ptr<FMI::Comm::IOState> state, FMI::Utils::peer_num rcpt_id) {
    std::string pairing = get_pairing_name(peer_id, rcpt_id, Utils::BLOCKING);
    check_socket(rcpt_id, pairing);
    int socketfd = sockets[Utils::BLOCKING][rcpt_id];

    ssize_t sent_total = 0;

    // Zero-length message? Send dummy byte
    if (state->request->len == 0) {
        char dummy = 0;
        while (true) {
            ssize_t sent = ::send(socketfd, &dummy, 1, 0);
            if (sent == 1) {
                if (state->callback) state->callback();
                state->callbackResult(Utils::SUCCESS, "Zero-length message sent with dummy byte.", state->context);
                return;
            } else if (sent == -1) {
                if (errno == EINTR) continue;
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    throw Utils::Timeout();
                }
                LOG(ERROR) << "Send error (dummy): " << strerror(errno);
                return;
            } else {
                state->callbackResult(Utils::DUMMY_SEND_FAILED, strerror(errno), state->context);
                return;
            }

        }

    }


    while (sent_total < state->request->len) {
        ssize_t sent = ::send(socketfd, state->request->buf.get() + sent_total,
                              state->request->len - sent_total, 0);

        if (sent == -1) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                throw Utils::Timeout(); // or use callbackResult if needed
            }
            LOG(ERROR) << "Send error: " << strerror(errno);
            return;
        }

        sent_total += sent;
    }

    if (state->callback) state->callback();
    state->callbackResult(Utils::SUCCESS, "Blocking send complete", state->context);
}

void FMI::Comm::Direct::send_object(std::shared_ptr<IOState> state, Utils::peer_num rcpt_id,
                                    Utils::Mode mode) {

    /*if (!blocking_init) {
        init_blocking_sockets();
    }*/

    if (mode == Utils::NONBLOCKING) {
        std::string pairing = get_pairing_name(peer_id, rcpt_id, Utils::NONBLOCKING);

        // Use full-duplex socket for both send/recv
        check_socket_nbx(rcpt_id, pairing);
        int socketfd = sockets[Utils::NONBLOCKING][rcpt_id];



        io_states[Utils::Operation::SEND][socketfd] = state;

        //if (checkSend(socketfd)) {
        //    handle_event(socketfd, io_states[Utils::SEND], Utils::SEND);
        //}

        // Zero-length message? Send dummy byte
        /*if (state.request.len == 0) {
            char dummy = 0;
            ssize_t sent = ::send(socketfd, &dummy, 1, 0);

            if (sent == 1) {
                if (state.callback) state.callback();
                state.callbackResult(Utils::SUCCESS, "Zero-length message sent with dummy byte.", state.context);
            } else if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                // Need to retry via epoll
                io_states[Utils::Operation::SEND][socketfd] = state;
                //add_epoll_event(socketfd, state);
            } else {
                state.callbackResult(Utils::DUMMY_SEND_FAILED, strerror(errno), state.context);
            }

            return;
        }



        // Normal message
        ssize_t processed = ::send(socketfd,
                                   state.request.buf.get() + state.processed,
                                   state.request.len - state.processed,
                                   0);

        if (processed > 0) {
            state.processed += processed;

            if (state.processed == state.request.len) {
                if (state.callback) state.callback();
                state.callbackResult(Utils::SUCCESS, "Send completed", state.context);
                return;
            }
        }

        // Still pending, register for epoll
        if (processed == -1 && (errno != EAGAIN && errno != EINTR)) {
            state.callbackResult(Utils::SEND_FAILED, strerror(errno), state.context);
            return;
        }*/

        // Save the state and try again via epoll
        //io_states[Utils::Operation::SEND][socketfd] = state;
        //add_epoll_event(socketfd, state);

    } else {
        send_object_blocking2(state, rcpt_id);
    }


}


void FMI::Comm::Direct::send_object(const std::shared_ptr<channel_data> buf, FMI::Utils::peer_num rcpt_id) {

    std::string pairing = get_pairing_name(peer_id, rcpt_id, Utils::BLOCKING);
    check_socket(rcpt_id, /*comm_name + std::to_string(peer_id) + "_"
                    + std::to_string(rcpt_id)*/pairing);
    //checkReceivePing(sockets[Utils::BLOCKING][rcpt_id], Utils::BLOCKING);
    long sent = ::send(sockets[Utils::BLOCKING][rcpt_id], buf->buf.get(), buf->len, 0);
    if (sent == -1) {
        if (errno == EAGAIN) {
            throw Utils::Timeout();
        }
        LOG(ERROR) << peer_id << ": Error when sending: " << strerror(errno) ;
    }
}

void FMI::Comm::Direct::recv_object_blocking2(std::shared_ptr<FMI::Comm::IOState> state, FMI::Utils::peer_num sender_id) {
    std::string pairing = get_pairing_name(peer_id, sender_id, Utils::BLOCKING);
    check_socket(sender_id, pairing);
    int sockfd = sockets[Utils::BLOCKING][sender_id];

    void *buffer = state->request->len == 0
                   ? static_cast<void *>(&state->dummy)
                   : state->request->buf.get() + state->processed;

    size_t remaining = state->request->len == 0 ? 1 : state->request->len - state->processed;

    while (remaining > 0) {
        ssize_t received = ::recv(sockfd, buffer, remaining, 0);

        if (received > 0) {
            state->processed += received;
            remaining -= received;
            buffer = static_cast<char *>(buffer) + received;

            LOG(INFO) << "processed receive bytes: " << state->processed << " of " << state->request->len;

            // 🔥 Check if full buffer received mid-loop
            if (state->processed == state->request->len) {
                // ✅ Protocol-level FIN check (YOUR LOGIC)
                if (state->request->len >= 8 * sizeof(int)) {
                    int *header = reinterpret_cast<int *>(state->request->buf.get());
                    if (header[0] == 0 && header[1] == CYLON_MSG_FIN) {
                        if (state->callback) state->callback();
                        state->callbackResult(Utils::SUCCESS, "Protocol FIN received", state->context);
                        return;
                    }
                }
                // ✅ Normal data completion
                if (state->callback) state->callback();
                state->callbackResult(Utils::SUCCESS, "Receive completed", state->context);
                return;
            }

        } else if (received == 0) {
            // Connection closed prematurely
            state->callbackResult(Utils::CONNECTION_CLOSED_BY_PEER,
                                 "Socket closed before full message or FIN was received", state->context);
            return;

        } else if (errno == EINTR) {
            continue; // Retry on interrupt

        } else {
            // Fatal recv error
            state->callbackResult(Utils::RECEIVE_FAILED, strerror(errno), state->context);
            return;
        }
    }

    // Handle zero-length message completion (if no loop body runs)
    if (state->request->len == 0) {
        if (state->callback) state->callback();
        state->callbackResult(Utils::SUCCESS, "Zero-length receive via dummy byte", state->context);
    }
}

void FMI::Comm::Direct::recv_object(std::shared_ptr<IOState> state, Utils::peer_num sender_id,
                                    Utils::Mode mode) {

    //if (!blocking_init) {
    //    init_blocking_sockets();
    //}

    if (mode == Utils::NONBLOCKING) {
        std::string pairing = get_pairing_name(peer_id, sender_id, Utils::NONBLOCKING);
        check_socket_nbx(sender_id, pairing);
        auto sender_socket = sockets[Utils::NONBLOCKING][sender_id];



        io_states[Utils::Operation::RECEIVE][sender_socket] = state;
        //if (checkRecv(sender_socket)) {
        //    handle_event(sender_socket, io_states[Utils::RECEIVE], Utils::RECEIVE);
        //}
        //add_epoll_event(sender_socket, state);
    } else {
        //check for ping message first


        recv_object_blocking2(state, sender_id);
    }
}

void FMI::Comm::Direct::recv_object(std::shared_ptr<channel_data> buf, FMI::Utils::peer_num sender_id) {



    std::string pairing = get_pairing_name(peer_id, sender_id, Utils::BLOCKING);
    check_socket(sender_id, /*comm_name + std::to_string(sender_id) + "_"
                                + std::to_string(peer_id)*/pairing);

    //checkReceivePing(sockets[Utils::BLOCKING][sender_id], Utils::BLOCKING);

    long received = ::recv(sockets[Utils::BLOCKING][sender_id], buf->buf.get(), buf->len, MSG_WAITALL);
    if (received == -1 || received < buf->len) {
        if (errno == EAGAIN) {
            throw Utils::Timeout();
        }
        LOG(ERROR) << peer_id << ": Error when receiving: " << strerror(errno);
    }
}

void FMI::Comm::Direct::check_socket(FMI::Utils::peer_num partner_id, std::string pair_name) {
    int current_try = 0;
    int max_tries = max_tcpunch_tries;
    while (current_try < max_tries) {
        if (sockets[Utils::BLOCKING].empty()) {
            sockets[Utils::BLOCKING] = std::vector<int>(num_peers, -1);
        }
        if (sockets[Utils::BLOCKING][partner_id] == -1) {
            try {
                sockets[Utils::BLOCKING][partner_id] = pair(pair_name, hostname, port,
                                                            holepunch_connect_to);
                LOG(INFO) << "Paired partnerId: " << partner_id << " to pair_name" << pair_name;
            } catch (Timeout e) {
                LOG(INFO) << "Socket pairing failed: " << std::string(e.what()) << " pairName: " << pair_name
                          << "partnerId: " << partner_id;

                //publish message so other end tries to connect
                remove_pair("remove_pair_" + pair_name, hostname, port,
                            holepunch_connect_to);

                /*if (redis_port > 0 && !redis_host.empty()) {
                    auto opts = sw::redis::ConnectionOptions{};
                    opts.host = redis_host;
                    opts.port = redis_port;
                    auto redis = std::make_shared<sw::redis::Redis>(opts);
                    std::string message = "from:" + std::to_string(peer_id) + ",to:" + std::to_string(partner_id);
                    redis->publish("fmi_connect", message);
                }*/

                current_try++;
                if (current_try == max_tries) {
                    throw Utils::Timeout();
                }
            } catch (ValidationFailure e) {
                LOG(INFO) << "Socket pairing validation failed: " << std::string(e.what()) << " pairName: " << pair_name
                          << "partnerId: " << partner_id;

                current_try++;
                if (current_try == max_tries) {
                    throw Utils::Timeout();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }

            struct timeval timeout;
            timeout.tv_sec = max_timeout / 1000;
            timeout.tv_usec = (max_timeout % 1000) * 1000;
            setsockopt(sockets[Utils::BLOCKING][partner_id], SOL_SOCKET, SO_RCVTIMEO,
                       (const char *) &timeout, sizeof timeout);
            setsockopt(sockets[Utils::BLOCKING][partner_id], SOL_SOCKET, SO_SNDTIMEO,
                       (const char *) &timeout, sizeof timeout);
            // Disable Nagle algorithm to avoid 40ms TCP ack delays
            int one = 1;
            int idle = 30;       // 30 seconds idle before starting keepalive probes
            int interval = 10;   // 10 seconds between keepalive probes
            int count = 3;       // Drop the connection after 3 failed probes
            //int bufsize = (1024 * 1024) * 100 ;
            // SOL_TCP not defined on macOS
#if !defined(SOL_TCP) && defined(IPPROTO_TCP)
#define SOL_TCP IPPROTO_TCP
#endif
            setsockopt(sockets[Utils::BLOCKING][partner_id], SOL_TCP, TCP_NODELAY,
                       &one, sizeof(one));

            setsockopt(sockets[Utils::BLOCKING][partner_id], SOL_SOCKET, SO_KEEPALIVE,
                       &one, sizeof(one));

            setsockopt(sockets[Utils::BLOCKING][partner_id], IPPROTO_TCP, TCP_KEEPIDLE,
                       &idle, sizeof(idle));
            setsockopt(sockets[Utils::BLOCKING][partner_id], IPPROTO_TCP, TCP_KEEPINTVL,
                       &interval, sizeof(interval));
            setsockopt(sockets[Utils::BLOCKING][partner_id], IPPROTO_TCP, TCP_KEEPCNT,
                       &count, sizeof(count));
            /*setsockopt(sockets[Utils::BLOCKING][partner_id], SOL_SOCKET, SO_SNDBUF,
                       &bufsize, sizeof(bufsize));
            setsockopt(sockets[Utils::BLOCKING][partner_id], SOL_SOCKET, SO_RCVBUF,
                       &bufsize, sizeof(bufsize));*/
            return;
        }
        current_try++;

    }
}



void FMI::Comm::Direct::check_timeouts(std::unordered_map<int, IOState> states) {
    /*auto now = std::chrono::steady_clock::now();
    for (auto it = states.begin(); it != states.end(); ) {
        if (now >= it->second.deadline) {
            it->second.callbackResult(Utils::NBX_TIMOUTOUT, "Operation timed out.", it->second.context);
            epoll_ctl(epoll_fd, EPOLL_CTL_DEL, it->first, nullptr);
            it = states.erase(it);
        } else {
            ++it;
        }
    }*/
}

FMI::Utils::EventProcessStatus
FMI::Comm::Direct::channel_event_progress(std::unordered_map<int, std::shared_ptr<IOState>> &states, Utils::Operation op) {
    if (states.empty()) {
        return FMI::Utils::EMPTY;
    }


    for (auto& [fd, state] : states) {

        if (op == Utils::SEND && checkSend(fd)) {
            handle_event(fd, states, op);
        } else if (op == Utils::RECEIVE && checkRecv2(fd)) {

            if (enable_ping) {
                checkReceivePing(fd, Utils::NONBLOCKING);
            }
            handle_event(fd, states, op);
        }

    }

    /*for (auto& [fd, state] : states) {
        if (op == Utils::SEND && checkSend(fd)) {
            handle_event(fd, states, op);
        } else if (op == Utils::RECEIVE && checkRecv2(fd)) {
            handle_event(fd, states, op);
        }
    }*/


    return FMI::Utils::PROCESSING;
}


FMI::Utils::EventProcessStatus FMI::Comm::Direct::channel_event_progress(Utils::Operation op) {

    if (op == Utils::DEFAULT) {
        FMI::Utils::EventProcessStatus status = FMI::Utils::EMPTY;
        for (auto& [operation, state] : io_states) {
            auto processStatus = channel_event_progress(state, op);
            if (processStatus != FMI::Utils::EMPTY) {
                status = processStatus;
            }
        }

        return status;
    } else {
        return channel_event_progress(io_states[op], op);
    }


}

void FMI::Comm::Direct::handle_event(int sockfd,
                                     std::unordered_map<int, std::shared_ptr<IOState>> &states,
                                     Utils::Operation op) const {


    if (op == Utils::SEND && /*(ev.events & EPOLLOUT) &&*/ states.count(sockfd)) {
        auto state = states[sockfd];
        // Zero-length message? Send dummy byte
        if (state->request->len == 0) {
            char dummy = 0;
            ssize_t sent = ::send(sockfd, &dummy, 1, 0);

            if (sent == 1) {
                if (state->callback) state->callback();
                state->callbackResult(Utils::SUCCESS, "Zero-length message sent with dummy byte.", state->context);

            } else if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                LOG(INFO) << "Send - retryable event: " << strerror(errno);
            } else {
                state->callbackResult(Utils::DUMMY_SEND_FAILED, strerror(errno), state->context);
            }

            return;
        }



        // Normal message
        ssize_t processed = ::send(sockfd,
                                   state->request->buf.get() + state->processed,
                                   state->request->len - state->processed,
                                   0);

        if (processed > 0) {
            state->processed += processed;

            if (state->processed == state->request->len) {

                if (state->request->len >= 8 * sizeof(int)) {
                    int *header = reinterpret_cast<int *>(state->request->buf.get());
                    if (header[1] == CYLON_MSG_FIN) {
                        LOG(INFO) << "Send: " << sockfd << " CYLON_MSG_FIN";

                    }
                }

                if (state->callback) state->callback();
                state->callbackResult(Utils::SUCCESS, "Send completed", state->context);

                return;
            }
        }

        // Still pending, register for epoll
        if (processed == -1 && (errno != EAGAIN && errno != EINTR)) {
            state->callbackResult(Utils::SEND_FAILED, strerror(errno), state->context);
            return;
        }

    }


    if (op == Utils::RECEIVE && states.count(sockfd)) {
        auto state = states[sockfd];

        void *buffer = state->request->len == 0
                       ? static_cast<void *>(&state->dummy)
                       : state->request->buf.get() + state->processed;

        size_t size = state->request->len == 0 ? 1 : state->request->len - state->processed;

        ssize_t received = ::recv(sockfd, buffer, size, 0);


        if (received > 0) {
            state->processed += received;

            LOG(INFO) << "processed receive bytes: " << state->processed << " of " << state->request->len;

            if (state->request->len == 0) {
                if (state->callback) state->callback();
                state->callbackResult(Utils::SUCCESS, "Zero-length receive via dummy byte", state->context);

            } else if (state->processed == state->request->len) {
                // Check for protocol-level FIN message
                if (state->request->len >= 8 * sizeof(int)) {
                    int *header = reinterpret_cast<int *>(state->request->buf.get());
                    if (header[1] == CYLON_MSG_FIN) {
                        LOG(INFO) << "recv: erasing " << sockfd << " CYLON_MSG_FIN";
                        if (state->callback) state->callback();
                        state->callbackResult(Utils::SUCCESS, "Protocol FIN received", state->context);
                        return;
                    }
                }


                if (state->callback) state->callback();
                state->callbackResult(Utils::SUCCESS, "Receive completed", state->context);
            }

        } else if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
            LOG(INFO) << "Recv - retryable event: " << strerror(errno);

        } else  {

            LOG(INFO) << "Recv: Error returned: " << strerror(errno) << " deleting socket: " << sockfd;
            state->callbackResult(Utils::RECEIVE_FAILED, strerror(errno), state->context);
            return;
        }
    }
}


void FMI::Comm::Direct::check_socket_nbx(FMI::Utils::peer_num partner_id, std::string pair_name) {


    if (sockets[Utils::NONBLOCKING].empty()) {
        sockets[Utils::NONBLOCKING] = std::vector<int>(num_peers, -1);
    }
    if (sockets[Utils::NONBLOCKING][partner_id] == -1) {

        int current_try = 0;
        int max_tries = max_tcpunch_tries;
        while (current_try < max_tries) {


        // 🔄 Use the original `pair()` function to establish the socket connection
        LOG(INFO) << "trying to pair partnerId: " << partner_id << " to pair_name" << pair_name;
        int result = pair(pair_name, hostname, port, max_timeout);
        
        if (result >= 0) {
            // Success
            sockets[Utils::NONBLOCKING][partner_id] = result;
            LOG(INFO) << "Paired partnerId: " << partner_id << " to pair_name" << pair_name;
        } else {
            // Handle different failure types
            const char* error_type = (result == -1) ? "timeout" : 
                                   (result == -2) ? "validation_failure" : "connection_failure";
            LOG(INFO) << "Socket pairing failed: " << error_type << " pairName: " << pair_name << "partnerId: " << partner_id;
            
            current_try++;
            if (current_try == max_tries) {
                LOG(INFO) << "maxtries for partnerId: " << partner_id << " to pair_name" << pair_name;
                return;
            }
            
            // Exponential backoff for retries
            int backoff_delay = 500 * (1 << std::min(current_try, 4)); // Cap at 8 seconds
            std::this_thread::sleep_for(std::chrono::milliseconds(backoff_delay));
            continue;
        }

        // ✅ Set the socket to non-blocking mode
        int flags = fcntl(sockets[Utils::NONBLOCKING][partner_id], F_GETFL, 0);
        if (flags == -1 || fcntl(sockets[Utils::NONBLOCKING][partner_id], F_SETFL, flags | O_NONBLOCK) == -1) {
            LOG(INFO) << "Failed to set non-blocking mode: " << std::string(strerror(errno));
            return;
        }

        // ✅ Configure socket timeouts
        struct timeval timeout;
        timeout.tv_sec = max_timeout / 1000;
        timeout.tv_usec = (max_timeout % 1000) * 1000;
        if (setsockopt(sockets[Utils::NONBLOCKING][partner_id], SOL_SOCKET, SO_RCVTIMEO,
                       &timeout, sizeof(timeout)) == -1) {
            LOG(INFO) << "Failed to set SO_RCVTIMEO: " << std::string(strerror(errno));
        }
        if (setsockopt(sockets[Utils::NONBLOCKING][partner_id], SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == -1) {
            LOG(INFO) << "Failed to set SO_SNDTIMEO: " + std::string(strerror(errno));
        }

        // ✅ Disable Nagle’s algorithm for low-latency communication
        int one = 1;
        if (setsockopt(sockets[Utils::NONBLOCKING][partner_id], IPPROTO_TCP,
                       TCP_NODELAY, &one, sizeof(one)) == -1) {
            LOG(INFO) << "Failed to set TCP_NODELAY: " << std::string(strerror(errno));
        }
        return;

        /*if (setsockopt(sockets[Utils::NONBLOCKING][partner_id], SOL_SOCKET, SO_KEEPALIVE,
                       &one, sizeof(one)) == -1) {
            LOG(INFO) << "Failed to set SO_KEEPALIVE: " << std::string(strerror(errno));
        }*/

        /*int idle = 30;       // 30 seconds idle before starting keepalive probes
        int interval = 10;   // 10 seconds between keepalive probes
        int count = 3;       // Drop the connection after 3 failed probes

        if (setsockopt(sockets[Utils::NONBLOCKING][partner_id], IPPROTO_TCP, TCP_KEEPIDLE,
                   &idle, sizeof(idle)) == -1) {
            LOG(INFO) << "Failed to set TCP_KEEPIDLE: " << std::string(strerror(errno));
        }

        if(setsockopt(sockets[Utils::NONBLOCKING][partner_id], IPPROTO_TCP, TCP_KEEPINTVL,
                   &interval, sizeof(interval)) == -1) {
            LOG(INFO) << "Failed to set TCP_KEEPINTVL: " << std::string(strerror(errno));
        }

        if(setsockopt(sockets[Utils::NONBLOCKING][partner_id], IPPROTO_TCP, TCP_KEEPCNT,
                   &count, sizeof(count)) == -1) {
            LOG(INFO) << "Failed to set TCP_KEEPCNT: " << std::string(strerror(errno));
        }*/
        }

    }


}

int FMI::Comm::Direct::getMaxTimeout() {
    return max_timeout;
}

bool FMI::Comm::Direct::checkSend(int fd) {

    pollfd pfd = { fd, POLLOUT, 0 };
    int poll_result = poll(&pfd, 1, 0);

    if (poll_result > 0) {
        return true;
    } else if (poll_result == 0) {
        // No events, not ready
        return false;
    } else {
        // poll() returned -1, check errno
        LOG(INFO) << "checkSend: poll() failed with errno " << errno << ": " << strerror(errno);
        return false;
    }
}

bool FMI::Comm::Direct::checkSend(FMI::Utils::peer_num dest, Utils::Mode mode) {

    //mapIfNotMapped(dest, mode);
    auto sockfd = sockets[mode][dest];
    return checkSend(sockfd);
}

bool FMI::Comm::Direct::checkRecv(int fd) {
    // Attempt to peek at the socket buffer without blocking
    char dummy;
    int peek = ::recv(fd, &dummy, 1, MSG_PEEK | MSG_DONTWAIT);

    if (peek > 0) {
        return true;  // ✅ Data is ready to read
    } else if (peek == 0) {
        // 🔒 Peer has closed the connection gracefully
        LOG(WARNING) << "checkReceive: socket " << fd << " closed the connection.";
        return false;
    } else {
        // ❌ Check error condition
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return false;  // ❌ No data available (non-blocking peek)
        } else {

            LOG(ERROR) << "checkReceive: recv error on socket " << fd << ": " << strerror(errno);

            return false;

        }
    }
}

bool FMI::Comm::Direct::checkRecv2(int fd) {
    pollfd pfd = { fd, POLLIN, 0 };
    int poll_result = poll(&pfd, 1, 0);

    if (poll_result > 0) {
        return true;  // ✅ Ready to read
    } else if (poll_result == 0) {
        return false; // ❌ Not ready yet
    } else {
        LOG(ERROR) << "checkRecv: poll() failed with errno " << errno << ": " << strerror(errno);
        return false;
    }
}

bool FMI::Comm::Direct::checkReceive(FMI::Utils::peer_num dest, Utils::Mode mode) {

    auto sockfd = sockets[mode][dest];
    return checkRecv(sockfd);

}

bool FMI::Comm::Direct::checkReceivePing(int sockfd, FMI::Utils::Mode mode) {

    if (sockfd == -1) {
        LOG(WARNING) << "checkIfOkToReceivePing: socket for peer " << peer_id << "Mode: " << ModeToString(mode) << " not initialized.";
        return false;
    }

    char peek_buf[16];
    ssize_t peeked = ::recv(sockfd, peek_buf, sizeof(peek_buf), MSG_PEEK | MSG_DONTWAIT);

    if (peeked == sizeof(peek_buf)) {
        int *header = reinterpret_cast<int *>(peek_buf);

        if (header[1] == CYLON_MSG_PING) {
            // Consume the ping fully
            char ping_buf[16];
            ssize_t consumed = ::recv(sockfd, ping_buf, sizeof(ping_buf), MSG_DONTWAIT);

            if (consumed == sizeof(ping_buf)) {
                LOG(INFO) << "[Direct::checkIfOkToReceivePing] Received and consumed PING from peer "
                            << peer_id << " Mode: " << ModeToString(mode);
            } else {
                LOG(WARNING) << "[Direct::checkIfOkToReceivePing] Failed to consume full PING from peer "
                             << peer_id << ", received: " << consumed << " Mode: " << ModeToString(mode);
            }

            // No reply needed (you're not sending PONG)
            return true;
        } else {
            // Some other message is present
            return false;
        }
    } else if (peeked == -1) {
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            LOG(WARNING) << "[Direct::checkIfOkToReceivePing] recv() failed: " << strerror(errno)
                    << " Mode: " << ModeToString(mode);
        }
    }



    return false; // No ping available or error
}




























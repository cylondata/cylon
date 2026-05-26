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

#ifndef CYLON_DIRECT_HPP
#define CYLON_DIRECT_HPP


#define CYLON_MSG_PING 9999
#define CYLON_MSG_PONG 9998

#include "PeerToPeer.hpp"
#include <cylon/thridparty/fmi/utils/Common.hpp>
#include <sys/epoll.h>
#include <thread>
#include <atomic>

namespace FMI::Comm {

    struct PingMessage {
        int magic = 0;
        int type = CYLON_MSG_PING;
        int reserved1 = 0;
        int reserved2 = 0;
    };

    //! Channel that uses the TCPunch TCP NAT Hole Punching Library for connection establishment.
    class Direct : public PeerToPeer {
    public:
        explicit Direct(const std::shared_ptr<FMI::Utils::Backends> &backend);

        virtual ~Direct();

        void init() override;

        int getMaxTimeout() override;

        void send_object(std::shared_ptr<channel_data> buf, Utils::peer_num rcpt_id) override;


        void send_object(std::shared_ptr<IOState> state, Utils::peer_num rcpt_id, Utils::Mode mode) override;

        void send_object_blocking2(std::shared_ptr<IOState> state, Utils::peer_num rcpt_id);

        void recv_object(std::shared_ptr<channel_data> buf, Utils::peer_num sender_id) override;

        void recv_object(std::shared_ptr<IOState> state, Utils::peer_num sender_id, Utils::Mode mode) override;

        void recv_object_blocking2(std::shared_ptr<IOState> state, Utils::peer_num sender_id);

        bool checkReceive(FMI::Utils::peer_num dest, Utils::Mode mode) override;

        bool checkSend(FMI::Utils::peer_num dest, Utils::Mode mode) override;


        Utils::EventProcessStatus channel_event_progress(Utils::Operation op) override;

        void start_holepunch_subscriber();

    private:
        //! Contains the socket file descriptor for the communication with the peers.
        std::unordered_map<Utils::Mode, std::vector<int>> sockets;

        std::vector<int> epoll_registered_fds;
        std::unordered_map<int, uint32_t> socket_event_map;

        std::string hostname;
        int port;
        bool resolve_host_dns;
        bool enable_ping;
        bool blocking_init = false;
        unsigned int max_timeout;
        Utils::Mode mode;


        std::unordered_map<Utils::Operation, std::unordered_map<int, std::shared_ptr<IOState>>> io_states;


        Utils::EventProcessStatus channel_event_progress(std::unordered_map<int, std::shared_ptr<IOState>> &states,
                                                         Utils::Operation op);

        //! Checks if connection with a peer partner_id is already established, otherwise establishes it using TCPunch.
        void check_socket(Utils::peer_num partner_id, std::string pair_name);

        void check_timeouts(std::unordered_map<int, IOState> states);

        void check_socket_nbx(Utils::peer_num partner_id, std::string pair_name);


        void handle_event(int socketfd,
                          std::unordered_map<int, std::shared_ptr<IOState>> &states,
                          Utils::Operation op) const;

        bool checkReceivePing(int sockeetfd, Utils::Mode mode);

        std::string get_pairing_name(Utils::peer_num a, Utils::peer_num b, Utils::Mode mode);

        bool checkSend(int fd);

        bool checkRecv(int fd);

        bool checkRecv2(int fd);

        void init_blocking_sockets();

        void start_ping_thread(Utils::Mode mode);

    };
}

#endif //CYLON_DIRECT_HPP

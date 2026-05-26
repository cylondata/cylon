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

#ifndef CYLON_FMI_CHANNEL_HPP
#define CYLON_FMI_CHANNEL_HPP

#include <unordered_map>
#include <queue>

#include "cylon/net/channel.hpp"
#include "cylon/thridparty/fmi/Communicator.hpp"
#include "fmi_operations.hpp"
#include <sw/redis++/redis++.h>
#include <iostream>
#include <thread>

using namespace sw::redis;

namespace cylon {

    namespace fmi {


        enum FMISendStatus {
            SEND_INIT = 0,
            SEND_LENGTH_POSTED = 1,
            SEND_POSTED = 2,
            SEND_FINISH = 3,
            SEND_DONE = 4
        };

        enum FMIReceiveStatus {
            RECEIVE_INIT = 0,
            RECEIVE_LENGTH_POSTED = 1,
            RECEIVE_POSTED = 2,
            RECEIVED_FIN = 3
        };

        enum PublishStatusType {
            SEND,
            RECEIVE
        };

        enum FMISendReceiveStatus {
            SENDING = 0,
            SEND_READY = 1,
            RECEIVING = 2,
            IDLE = 3,
            INVALID = 4
        };


        /**
        * Keep track about the length buffer to receive the length first
        */
        struct PendingSend {
            //  we allow upto 8 ints for the header
            int headerBuf[CYLON_CHANNEL_HEADER_SIZE]{};
            // segments of data to be sent
            std::queue<std::shared_ptr<CylonRequest>> pendingData{};

            FMISendStatus status = SEND_INIT;

            FMISendReceiveStatus sendReceiveStatus = IDLE;

            // the current send, if it is a actual send
            std::shared_ptr<CylonRequest> currentSend{};

            // UCX context - For tracking the progress of the message
            FMI::Utils::fmiContext *context = nullptr;

        };

        struct PendingReceive {
            // we allow upto 8 integer header
            int headerBuf[CYLON_CHANNEL_HEADER_SIZE]{};
            int receiveId{};
            // Buffers are untyped: they simply denote a physical memory
            // area regardless of its intended meaning or interpretation.
            std::shared_ptr<Buffer> data{};
            int length{};
            FMIReceiveStatus status = RECEIVE_INIT;

            // FMI context - For tracking the progress of the message
            FMI::Utils::fmiContext *context = nullptr;
        };


        class FMIChannel : public Channel {


        public:

            /**
            * Initialize the channel
            *
            * @param receives receive from these ranks
            */
            void init(int edge,
                      const std::vector<int> &receives,
                      const std::vector<int> &sendIds,
                      ChannelReceiveCallback *rcv,
                      ChannelSendCallback *send,
                      Allocator *alloc) override;

            /**
            * Send the message to the target.
            *
            * @param request the request
            * @return true if accepted
            */
            int send(std::shared_ptr<CylonRequest> request) override;

            /**
            * Send the message to the target.
            *
            * @param request the request
            * @return true if accepted
            */
            int sendFin(std::shared_ptr<CylonRequest> request) override;

            /**
             * This method, will send the messages, It will first send a message with length and then
             */
            void progressSends() override;

            void progressSendTo(int peer_id);

            /*void progressSendsBlocking();*/

            /**
             * Progress the pending receivers
             */
            void progressReceives() override;

            /*void progressReceivesBlocking();*/

            void progressReceiveFrom(int peer_id);

            void close() override;

            explicit FMIChannel(std::shared_ptr<FMI::Communicator> com, FMI::Utils::Mode mode,
                                std::string redis_host, int redis_port, std::string redis_namespace);

            void notifyCompleted() override;

        private:
            // keep track of the length buffers for each receiver
            std::unordered_map<int, PendingSend *> sends;
            // keep track of the posted receives
            std::unordered_map<int, PendingReceive *> pendingReceives;
            // we got finish requests
            std::unordered_map<int, std::shared_ptr<CylonRequest>> finishRequests;
            //send turn for blocking communication
            // receive callback function
            ChannelReceiveCallback *rcv_fn;
            // send complete callback function
            ChannelSendCallback *send_comp_fn;
            // allocator
            Allocator *allocator;
            // mpi rank
            int rank;
            // mpi world size
            int worldSize;


            std::shared_ptr<FMI::Communicator> communicator;

            FMI::Utils::Mode mode;

            std::string redis_host;

            int redis_port;

            std::string redis_namespace;



            std::shared_ptr<sw::redis::Redis> redis;

            std::string global_peer_lock;


            //Thread support
            std::unordered_map<int, std::thread> commThreads;
            std::atomic<bool> commStarted{false};
            std::atomic<bool> shutdown{false};
            std::unordered_map<int, std::shared_ptr<std::mutex>> send_mutex_;
            std::unordered_map<int, std::shared_ptr<std::mutex>> recv_mutex_;
            std::shared_ptr<std::mutex> getSendMutex(int peer_id);
            std::shared_ptr<std::mutex> getRecvMutex(int peer_id);

            std::mutex send_mutex;
            std::mutex recv_mutex;


            /**
             * UCX Receive
             * Modeled after the IRECV function of MPI
             * @param [out] buffer - Pointer to the output buffer
             * @param [in] count - Size of the receiving data
             * @param [in] sender - MPI id of the sender
             * @param [out] ctx - ucx::ucxContext object, used for tracking the progress of the request
             * @return Cylon Status
             */
            template<typename T>
            Status FMI_Irecv(FMI::Comm::Data<T> &buf,
                             int sender,
                             FMI::Utils::fmiContext* ctx);

            /**
             * UCX Send
             * Modeled after the ISEND function of MPI
             * @param [out] buffer - Pointer to the buffer to send
             * @param [in] count - Size of the receiving data
             * @param [in] ep - Endpoint to send the data to
             * @param [out] request - UCX Context object
             *                        Used for tracking the progress of the request
             * @return Cylon Status
             */
            template<typename T>
            Status FMI_Isend(FMI::Comm::Data<T> &buf,
                             int source,
                             FMI::Utils::fmiContext* request) const;

            /**
             * Send finish request
             * @param x the target, pendingSend pair
             */
            void sendFinishHeader(const std::pair<const int, PendingSend *> &x) const;

            void sendFinishHeader(int target, PendingSend *ps);

            /**
             * Send the length
             * @param x the target, pendingSend pair
             */
            void sendHeader(const std::pair<const int, PendingSend *> &x) const;

            void sendHeader(int target, PendingSend *ps);

            void sendHeaderLocal(PendingSend * pend_send);
            void sendFinishHeaderLocal(PendingSend * pend_send);
            void progressSendsLocal(PendingSend * pend_send);

            bool acquire_lock(const std::string &lock_key,
                              const std::string &lock_value, int ttl_ms);

            void release_lock(const std::string &lock_key,
                              const std::string &lock_value);

            std::string generate_unique_id();

            void publishStatus(int rank, int peer_id, FMISendReceiveStatus sendRecvStatus,
                               PublishStatusType publishStatus);

            void startCommunicationThreads();

            std::string get_shared_lock_key(int a, int b);



        };
    }


}
#endif //CYLON_FMI_CHANNEL_HPP

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

#ifndef CYLON_COMMON_HPP
#define CYLON_COMMON_HPP

#include <exception>
namespace FMI::Utils {
    //! Type for peer IDs / numbers
    using peer_num = int;

    //! Custom exception that is thrown on timeouts
    struct Timeout : public std::exception {
        [[nodiscard]] const char * what () const noexcept {
            return "Timeout was reached";
        }
    };

    //! Set by the client, controls the optimization goal of the Channel Policy
    enum Hint {
        fast, cheap
    };

    enum NbxStatus {
        SUCCESS,
        SEND_FAILED,
        RECEIVE_FAILED,
        DUMMY_SEND_FAILED,
        CONNECTION_CLOSED_BY_PEER,
        SOCKET_CREATE_FAILED,
        TCP_NODELAY_FAILED,
        FCNTL_GET_FAILED,
        FCNTL_SET_FAILED,
        ADD_EVENT_FAILED,
        EPOLL_WAIT_FAILED,
        SOCKET_PAIR_FAILED,
        SOCKET_SET_SO_RCVTIMEO_FAILED,
        SOCKET_SET_SO_SNDTIMEO_FAILED,
        SOCKET_SET_TCP_NODELAY_FAILED,
        SOCKET_SET_NONBLOCKING_FAILED,
        NBX_TIMOUTOUT
    };

    enum EventProcessStatus {
        PROCESSING,
        EMPTY,
        NOOP
    };

    enum Mode {
        BLOCKING, NONBLOCKING
    };

    //! List of currently supported collectives
    enum Operation {
        SEND = 0x001,
        RECEIVE = 0x002,
        BCAST = 0x003,
        BARRIER = 0x004,
        ALLGATHER = 0x005,
        ALLGATHERV = 0x006,
        GATHER = 0x007,
        GATHERV = 0x008,
        SCATTER = 0x009,
        REDUCE = 0x010,
        ALLREDUCE = 0x011,
        SCAN = 0x012,
        DEFAULT = 0x020
    };

    //! All the information about an operation, passed to the Channel Policy for its decision on which channel to use.
    struct OperationInfo {
        Operation op;
        std::size_t data_size;
        bool left_to_right = false;
    };

    /**
        * Hold the completion status of a communication
        * completed - if completed show 1, else 0
        */
    struct fmiContext {
        int completed;
    };








}



#endif //CYLON_COMMON_HPP

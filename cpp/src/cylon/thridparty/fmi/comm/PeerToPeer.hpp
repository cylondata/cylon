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

#ifndef CYLON_PEERTOPEER_HPP
#define CYLON_PEERTOPEER_HPP

#include <utility>

#include "Channel.hpp"
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <chrono>


namespace FMI::Comm {
    //! Peer-To-Peer channel type
    /*!
     * This class provides optimized collectives for channels where clients can address each other directly and defines the interface that these channels need to implement.
     */

    struct GatherVData {
        std::size_t buf_len;
        const std::shared_ptr<channel_data> recvbuf;
        const std::vector<int32_t> &displs;
        const std::shared_ptr<channel_data> buffer;
        Utils::peer_num real_src;
    };

    struct GatherData {
        std::size_t buf_len;
        const channel_data &recvbuf;
        const channel_data &buffer;
        Utils::peer_num real_src;
        std::size_t single_buffer_size;
    };





    struct IOState {
        std::shared_ptr<channel_data> request;
        size_t processed{};
        Utils::Operation operation = Utils::DEFAULT;
        Utils::fmiContext * context = nullptr;
        char dummy = 0;
        std::function<void(FMI::Utils::NbxStatus, const std::string&, FMI::Utils::fmiContext *)> callbackResult;

        std::function<void()> callback = nullptr;  // Store function with bound arguments

        void setRequest(const std::shared_ptr<channel_data> cdata) {
            request = cdata;
        }



        template <typename Func, typename... Args>
        void setCallback(Func&& func, Args&&... args) {
            callback = std::bind(std::forward<Func>(func), std::forward<Args>(args)...);
        }

        std::chrono::steady_clock::time_point deadline;

        IOState() = default;



    };

    class PeerToPeer : public Channel {
    public:
        void send(std::shared_ptr<channel_data> buf, FMI::Utils::peer_num dest) override;


        void send(std::shared_ptr<channel_data> buf, FMI::Utils::peer_num dest,
                  FMI::Utils::fmiContext *context,
                  FMI::Utils::Mode mode,
                  std::function<void(FMI::Utils::NbxStatus, const std::string &,
                                     FMI::Utils::fmiContext *)> callback) override;

        void send(FMI::Utils::peer_num src,
                      Utils::Mode mode,
                      std::shared_ptr<IOState>  state);

        void recv(std::shared_ptr<channel_data> buf, FMI::Utils::peer_num src) override;


        void recv(FMI::Utils::peer_num src,
                      Utils::Mode mode,
                      std::shared_ptr<IOState> state);

        void recv(std::shared_ptr<channel_data> buf, FMI::Utils::peer_num src,
                  Utils::fmiContext * context,
                  Utils::Mode mode,
                  std::function<void(FMI::Utils::NbxStatus, const std::string&, FMI::Utils::fmiContext *)> callback);

        //! Binomial tree broadcast implementation
        void bcast(std::shared_ptr<channel_data> buf, FMI::Utils::peer_num root, Utils::Mode mode,
                   std::function<void(FMI::Utils::NbxStatus, const std::string &, FMI::Utils::fmiContext *)> callback) override;

        //! Calls allreduce with a (associative and commutative) NOP operation
        void barrier() override;

        //! Binomial tree gather.
        /*!
         * In the beginning, the needed buffer size (largest value that this peer will receive) is determined and a buffer is allocated.
         * If the ID of the root is not 0, we cannot necessarily receive all values directly in recvbuf because we need to wrap around (e.g., when we get from peer N - 1 the values for N - 1, 0, and 1).
         * This is solved by allocating a temporary buffer and copying the values.
         */
        void gather(const std::shared_ptr<channel_data> sendbuf,
                    std::shared_ptr<channel_data> recvbuf, FMI::Utils::peer_num root) override;


        void gatherv(const std::shared_ptr<channel_data> sendbuf,
                     std::shared_ptr<channel_data> recvbuf, FMI::Utils::peer_num root,
                     const std::vector<int32_t> &recvcounts,
                     const std::vector<int32_t> &displs,
                     Utils::Mode mode,
                     std::function<void(FMI::Utils::NbxStatus, const std::string&, FMI::Utils::fmiContext *)> callback) override;

        void
        allgather(const std::shared_ptr<channel_data> sendbuf,
                  std::shared_ptr<channel_data> recvbuf, FMI::Utils::peer_num root,
                  Utils::Mode mode,
                  std::function<void(FMI::Utils::NbxStatus, const std::string &, FMI::Utils::fmiContext *)> callback) override;


        void allgatherv(const std::shared_ptr<channel_data> sendbuf,
                        std::shared_ptr<channel_data> recvbuf, Utils::peer_num root,
                            const std::vector<int32_t> &recvcounts,
                            const std::vector<int32_t> &displs,
                            Utils::Mode mode,
                            std::function<void(FMI::Utils::NbxStatus, const std::string&, FMI::Utils::fmiContext *)> callback) override;
        //! Binomial tree scatter
        /*!
         * Similarly to gather, the root may need to send values from its sendbuf that is not consecutive when its ID is not 0, which is solved with a temporary buffer.
         */
        void scatter(const std::shared_ptr<channel_data> sendbuf,
                     std::shared_ptr<channel_data> recvbuf, FMI::Utils::peer_num root) override;

        //! Calls reduce_no_order for associative and commutative functions, reduce_ltr otherwise
        void reduce(const std::shared_ptr<channel_data> sendbuf,
                    std::shared_ptr<channel_data> recvbuf, FMI::Utils::peer_num root, raw_function f) override;

        //! For associative and commutative functions, allreduce_no_order is called. Otherwise, reduce followed by bcast is used.
        void allreduce(const std::shared_ptr<channel_data> sendbuf,
                       std::shared_ptr<channel_data> recvbuf, raw_function f) override;

        //! For associative and commutative functions, scan_no_order is called. Otherwise, scan_ltr is called
        void scan(const std::shared_ptr<channel_data> sendbuf,
                  std::shared_ptr<channel_data> recvbuf, raw_function f) override;

        //! Send an object to peer with ID peer_id. Needs to be implemented by the channels.
        virtual void send_object(const std::shared_ptr<channel_data> buf, Utils::peer_num peer_id) = 0;

        //! Send an object to peer with ID peer_id. Needs to be implemented by the channels(non-blocking).

        virtual void send_object(std::shared_ptr<IOState> state, Utils::peer_num peer_id, Utils::Mode mode) = 0;

        //! Receive an object from peer with ID peer_id. Needs to be implemented by the channels.
        virtual void recv_object(const std::shared_ptr<channel_data> buf, Utils::peer_num peer_id) = 0;

        //! Receive an object from peer with ID peer_id. Needs to be implemented by the channels (non-blocking).
        virtual void recv_object(std::shared_ptr<IOState> state, Utils::peer_num peer_id, Utils::Mode mode) = 0;


        Utils::EventProcessStatus channel_event_progress(Utils::Operation op) override;

    protected:
        //! Reduction with left-to-right evaluation, gather followed by a function evaluation on the root peer.
        void reduce_ltr(const std::shared_ptr<channel_data> sendbuf,
                        std::shared_ptr<channel_data> recvbuf,
                        FMI::Utils::peer_num root, const raw_function& f);

        //! Binomial tree reduction where all peers apply the function in every step.
        void reduce_no_order(const std::shared_ptr<channel_data> sendbuf,
                             const std::shared_ptr<channel_data> recvbuf,
                             FMI::Utils::peer_num root, const raw_function& f);

        //! Recursive doubling allreduce implementation. When num_peers is not a power of two, there is an additional message in the beginning and end for every peer where they send their value / receive the reduced value.
        void allreduce_no_order(const std::shared_ptr<channel_data> sendbuf,
                                const std::shared_ptr<channel_data> recvbuf, const raw_function& f);

        //! Linear function application / sending
        void scan_ltr(const std::shared_ptr<channel_data> sendbuf,
                      const std::shared_ptr<channel_data> recvbuf, const raw_function& f);

        //! Binomial tree with up- and down-phase
        void scan_no_order(const std::shared_ptr<channel_data> sendbuf,
                           const std::shared_ptr<channel_data> recvbuf, const raw_function& f);

    private:
        //! Allows to implement all collectives as if root were 0
        /*!
         * Transforms peer IDs such that the the user-provided root ID has a transformed ID of 0.
         * Makes the implementation of many collectives easier, because they only need to be implemented for the case with root = 0, when the transformation is used in the appropriate places
         * @param id ID to transform
         * @param root User-chosen root ID
         * @param forward Forward (root -> 0) or backward (0 -> root) transformation
         * @return transformed peer ID
         */
        Utils::peer_num transform_peer_id(Utils::peer_num id, Utils::peer_num root, bool forward);



    };
}



#endif //CYLON_PEERTOPEER_HPP

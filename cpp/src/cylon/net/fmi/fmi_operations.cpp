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

#include "fmi_operations.hpp"
#include "cylon/util/macros.hpp"
#include <glog/logging.h>
#include <optional>




    namespace cylon::fmi {

        template<typename T>
        FMI::Utils::Function<T> get_function(cylon::net::ReduceOp reduce_op) {
            switch (reduce_op) {
                case cylon::net::SUM: {
                    return FMI::Utils::Function<T>([](T a, T b) { return a + b; }, true, true);
                }
                case cylon::net::MIN: {
                    return FMI::Utils::Function<T>([](T a, T b) { return std::min(a, b); }, true, true);
                }
                case cylon::net::MAX: {
                    return FMI::Utils::Function<T>([](T a, T b) { return std::max(a, b); }, true, true);
                }
                case cylon::net::PROD: {
                    return FMI::Utils::Function<T>([](T a, T b) { return a * b; }, true, true);
                }
                default:
                    return FMI::Utils::Function<T>();
            }
        }

        std::string NbxStatusToString(FMI::Utils::NbxStatus status) {
            switch (status) {
                case FMI::Utils::NbxStatus::SUCCESS:
                    return "SUCCESS";
                case FMI::Utils::NbxStatus::CONNECTION_CLOSED_BY_PEER:
                    return "CONNECTION_CLOSED_BY_PEER";
                case FMI::Utils::NbxStatus::SOCKET_CREATE_FAILED:
                    return "SOCKET_CREATE_FAILED";
                case FMI::Utils::NbxStatus::TCP_NODELAY_FAILED:
                    return "TCP_NODELAY_FAILED";
                case FMI::Utils::NbxStatus::FCNTL_GET_FAILED:
                    return "FCNTL_GET_FAILED";
                case FMI::Utils::NbxStatus::FCNTL_SET_FAILED:
                    return "FCNTL_SET_FAILED";
                case FMI::Utils::NbxStatus::ADD_EVENT_FAILED:
                    return "ADD_EVENT_FAILED";
                case FMI::Utils::NbxStatus::EPOLL_WAIT_FAILED:
                    return "EPOLL_WAIT_FAILED";
                case FMI::Utils::NbxStatus::SOCKET_PAIR_FAILED:
                    return "SOCKET_PAIR_FAILED";
                case FMI::Utils::NbxStatus::SOCKET_SET_SO_RCVTIMEO_FAILED:
                    return "SOCKET_SET_SO_RCVTIMEO_FAILED";
                case FMI::Utils::NbxStatus::SOCKET_SET_SO_SNDTIMEO_FAILED:
                    return "SOCKET_SET_SO_SNDTIMEO_FAILED";
                case FMI::Utils::NbxStatus::SOCKET_SET_TCP_NODELAY_FAILED:
                    return "SOCKET_SET_TCP_NODELAY_FAILED";
                case FMI::Utils::NbxStatus::SOCKET_SET_NONBLOCKING_FAILED:
                    return "SOCKET_SET_NONBLOCKING_FAILED";
                default:
                    return "UNKNOWN_STATUS";
            }
        }

        void FmiTableAllgatherImpl::Init(int num_buffers) {
            CYLON_UNUSED(num_buffers);
        }

        Status FmiTableAllgatherImpl::AllgatherBufferSizes(const int32_t *send_data, int num_buffers,
                                                           int32_t *rcv_data) const {

            auto send_data_byte_size = num_buffers * sizeof(int32_t);
            auto send_void_ptr = const_cast<void *>(static_cast<const void *>(send_data));
            FMI::Comm::Data<void *> send_void_data(send_void_ptr, send_data_byte_size,
                                                   FMI::Comm::noop_deleter);
            auto recv_data_byte_size = comm_ptr_->getNumPeers() * num_buffers * sizeof(int32_t);
            auto recv_void_ptr = const_cast<void *>(static_cast<const void *>(rcv_data));
            FMI::Comm::Data<void *> recv_void_data(recv_void_ptr, recv_data_byte_size,
                                                   FMI::Comm::noop_deleter);
            comm_ptr_->allgather(send_void_data, recv_void_data, 0);
            return Status::OK();
        }

        Status FmiTableAllgatherImpl::IallgatherBufferData(int buf_idx, const uint8_t *send_data, int32_t send_count,
                                                           uint8_t *recv_data, const std::vector<int32_t> &recv_count,
                                                           const std::vector<int32_t> &displacements) {

            auto send_data_byte_size = send_count * sizeof(uint8_t);
            auto send_void_ptr = const_cast<void *>(static_cast<const void *>(send_data));
            FMI::Comm::Data<void *> send_void_data(send_void_ptr, send_data_byte_size,
                                                   FMI::Comm::noop_deleter);

            std::size_t total_recv_size = 0;
            for (size_t i = 0; i < comm_ptr_->getNumPeers(); i++) {
                total_recv_size += recv_count[i];
            }

            auto recv_data_byte_size = total_recv_size * sizeof(uint8_t);
            auto recv_void_ptr = const_cast<void *>(static_cast<const void *>(recv_data));
            FMI::Comm::Data<void *> recv_void_data(recv_void_ptr, recv_data_byte_size,
                                                   FMI::Comm::noop_deleter);
            comm_ptr_->allgatherv(send_void_data, recv_void_data, 0, recv_count,
                                         displacements, mode_,
                                         [](FMI::Utils::NbxStatus status, const std::string &msg,
                                            FMI::Utils::fmiContext * ctx) {
                                             CYLON_UNUSED(ctx);
                                             if (status != FMI::Utils::SUCCESS) {
                                                 LOG(ERROR) << "FMI IallgatherBufferData status: "
                                                            << NbxStatusToString(status) << " msg: " << msg;
                                             }
                                         });

            return Status::OK();
        }

        Status FmiTableAllgatherImpl::WaitAll(int num_buffers) {
            CYLON_UNUSED(num_buffers);
            if (mode_ == FMI::Utils::NONBLOCKING) {
                while (comm_ptr_->communicator_event_progress(FMI::Utils::Operation::DEFAULT) ==
                       FMI::Utils::EventProcessStatus::PROCESSING) {}
            }

            return Status::OK();
        }

        void FmiTableGatherImpl::Init(int num_buffers) {
            CYLON_UNUSED(num_buffers);
        }

        Status FmiTableGatherImpl::GatherBufferSizes(const int32_t *send_data, int num_buffers, int32_t *rcv_data,
                                                     int gather_root) const {

            auto send_data_byte_size = num_buffers * sizeof(int32_t);
            auto send_void_ptr = const_cast<void *>(static_cast<const void *>(send_data));
            FMI::Comm::Data<void *> send_void_data(send_void_ptr, send_data_byte_size,
                                                   FMI::Comm::noop_deleter);
            auto recv_data_byte_size = comm_ptr_->getNumPeers() * num_buffers * sizeof(int32_t);
            auto recv_void_ptr = const_cast<void *>(static_cast<const void *>(rcv_data));
            FMI::Comm::Data<void *> recv_void_data(recv_void_ptr, recv_data_byte_size,
                                                   FMI::Comm::noop_deleter);
            comm_ptr_->gather(send_void_data, recv_void_data, 0);
            return Status::OK();
        }

        Status FmiTableGatherImpl::IgatherBufferData(int buf_idx, const uint8_t *send_data, int32_t send_count,
                                                     uint8_t *recv_data, const std::vector<int32_t> &recv_count,
                                                     const std::vector<int32_t> &displacements, int gather_root) {
            auto send_data_byte_size = send_count * sizeof(uint8_t);
            auto send_void_ptr = const_cast<void *>(static_cast<const void *>(send_data));
            FMI::Comm::Data<void *> send_void_data(send_void_ptr, send_data_byte_size,
                                                   FMI::Comm::noop_deleter);

            std::size_t total_recv_size = 0;
            for (size_t i = 0; i < comm_ptr_->getNumPeers(); i++) {
                total_recv_size += recv_count[i];
            }

            auto recv_data_byte_size = total_recv_size * sizeof(uint8_t);
            auto recv_void_ptr = const_cast<void *>(static_cast<const void *>(recv_data));
            FMI::Comm::Data<void *> recv_void_data(recv_void_ptr, recv_data_byte_size,
                                                   FMI::Comm::noop_deleter);
            comm_ptr_->gatherv(send_void_data, recv_void_data, 0, recv_count,
                                      displacements, mode_,
                                      [](FMI::Utils::NbxStatus status, const std::string &msg,
                                         FMI::Utils::fmiContext * ctx) {
                                          CYLON_UNUSED(ctx);
                                          if (status != FMI::Utils::SUCCESS) {
                                              LOG(ERROR) << "FMI IgatherBufferData status: "
                                                         << NbxStatusToString(status)
                                                         << " msg: " << msg;
                                          }
                                      });

            return Status::OK();
        }

        Status FmiTableGatherImpl::WaitAll(int num_buffers) {
            CYLON_UNUSED(num_buffers);
            if (mode_ == FMI::Utils::NONBLOCKING) {
                while (comm_ptr_->communicator_event_progress(FMI::Utils::Operation::DEFAULT) ==
                       FMI::Utils::EventProcessStatus::PROCESSING) {}
            }

            return Status::OK();
        }

        void FmiTableBcastImpl::Init(int32_t num_buffers) {
            CYLON_UNUSED(num_buffers);
        }

        Status FmiTableBcastImpl::BcastBufferSizes(int32_t *buffer, int32_t count, int32_t bcast_root) const {

            auto data_byte_size = count * sizeof(int32_t);
            auto send_void_ptr = const_cast<void *>(static_cast<const void *>(buffer));
            FMI::Comm::Data<void *> send_void_data(send_void_ptr, data_byte_size,
                                                   FMI::Comm::noop_deleter);
            comm_ptr_->bcast(send_void_data, bcast_root);
            return Status::OK();
        }

        Status FmiTableBcastImpl::BcastBufferData(uint8_t *buf_data, int32_t send_count, int32_t bcast_root) const {
            auto data_byte_size = send_count * sizeof(int32_t);
            auto send_void_ptr = const_cast<void *>(static_cast<const void *>(buf_data));
            FMI::Comm::Data<void *> send_void_data(send_void_ptr, data_byte_size,
                                                   FMI::Comm::noop_deleter);
            comm_ptr_->bcast(send_void_data, bcast_root);
            return Status::OK();
        }

        Status FmiTableBcastImpl::IbcastBufferData(int32_t buf_idx, uint8_t *buf_data, int32_t send_count,
                                                   int32_t bcast_root) {
            auto data_byte_size = send_count * sizeof(int32_t);
            auto send_void_ptr = const_cast<void *>(static_cast<const void *>(buf_data));
            FMI::Comm::Data<void *> send_void_data(send_void_ptr, data_byte_size,
                                                   FMI::Comm::noop_deleter);
            comm_ptr_->bcast(send_void_data, bcast_root, mode_,
                                    [](FMI::Utils::NbxStatus status, const std::string &msg,
                                       FMI::Utils::fmiContext * ctx) {
                                        CYLON_UNUSED(ctx);
                                        if (status != FMI::Utils::SUCCESS) {
                                            LOG(ERROR) << "FMI IbcastBufferData status: " << NbxStatusToString(status)
                                                       << " msg: " << msg;
                                        }
                                    });
            return Status::OK();
        }

        Status FmiTableBcastImpl::WaitAll(int32_t num_buffers) {
            CYLON_UNUSED(num_buffers);
            if (mode_ == FMI::Utils::NONBLOCKING) {
                while (comm_ptr_->communicator_event_progress(FMI::Utils::Operation::DEFAULT) ==
                       FMI::Utils::EventProcessStatus::PROCESSING) {}
            }

            return Status::OK();
        }


        template<typename T>
        Status all_reduce_buffer(const std::shared_ptr<FMI::Communicator> &comm_ptr,
                                 const void *send_buf,
                                 void *rcv_buf,
                                 int count,
                                 net::ReduceOp reduce_op) {


            auto func = get_function<T>(reduce_op);
            if (!func.isValid()) {
                return {Code::Invalid, "Unsupported reduction operator " + std::to_string(reduce_op)};
            }

            auto data_byte_size = count * sizeof(T);
            auto send_void_ptr = const_cast<void *>(static_cast<const void *>(send_buf));
            FMI::Comm::Data<void *> send_void_data(send_void_ptr, data_byte_size,
                                                   FMI::Comm::noop_deleter);
            auto recv_void_ptr = const_cast<void *>(static_cast<const void *>(rcv_buf));
            FMI::Comm::Data<void *> recv_void_data(recv_void_ptr, data_byte_size,
                                                   FMI::Comm::noop_deleter);

            auto f = FMI::convert_to_raw_function(func, data_byte_size);

            comm_ptr->allreduce(send_void_data, recv_void_data,
                                func.commutative, func.associative, f);

            return Status::OK();
        }


        Status FmiAllReduceImpl::AllReduceBuffer(const void *send_buf, void *rcv_buf, int count,
                                                 const std::shared_ptr<DataType> &data_type,
                                                 net::ReduceOp reduce_op) const {

            switch (data_type->getType()) {
                case Type::BOOL:
                    break;
                case Type::UINT8:
                    return all_reduce_buffer<uint8_t>(comm_ptr_,
                                                      send_buf,
                                                      rcv_buf,
                                                      count,
                                                      reduce_op);

                case Type::INT8:
                    return all_reduce_buffer<int8_t>(comm_ptr_,
                                                     send_buf,
                                                     rcv_buf,
                                                     count,
                                                     reduce_op);
                case Type::UINT16:
                    return all_reduce_buffer<uint16_t>(comm_ptr_,
                                                       send_buf,
                                                       rcv_buf,
                                                       count,
                                                       reduce_op);
                case Type::INT16:
                    return all_reduce_buffer<int16_t>(comm_ptr_,
                                                      send_buf,
                                                      rcv_buf,
                                                      count,
                                                      reduce_op);
                case Type::UINT32:
                    return all_reduce_buffer<uint32_t>(comm_ptr_,
                                                       send_buf,
                                                       rcv_buf,
                                                       count,
                                                       reduce_op);
                case Type::INT32:
                    return all_reduce_buffer<int32_t>(comm_ptr_,
                                                      send_buf,
                                                      rcv_buf,
                                                      count, reduce_op);
                case Type::UINT64:
                    return all_reduce_buffer<uint64_t>(comm_ptr_,
                                                       send_buf,
                                                       rcv_buf,
                                                       count,
                                                       reduce_op);
                case Type::INT64:
                    return all_reduce_buffer<int64_t>(comm_ptr_,
                                                      send_buf,
                                                      rcv_buf,
                                                      count,
                                                      reduce_op);
                case Type::HALF_FLOAT:
                    break;
                case Type::FLOAT:
                    return all_reduce_buffer<float>(comm_ptr_,
                                                    send_buf,
                                                    rcv_buf,
                                                    count,
                                                    reduce_op);
                case Type::DOUBLE:
                    return all_reduce_buffer<double>(comm_ptr_,
                                                     send_buf,
                                                     rcv_buf,
                                                     count,
                                                     reduce_op);
                case Type::DATE32:
                case Type::TIME32:
                    return all_reduce_buffer<uint32_t>(comm_ptr_,
                                                       send_buf,
                                                       rcv_buf,
                                                       count,
                                                       reduce_op);
                case Type::DATE64:
                case Type::TIMESTAMP:
                case Type::TIME64:
                    return all_reduce_buffer<uint64_t>(comm_ptr_,
                                                       send_buf,
                                                       rcv_buf,
                                                       count,
                                                       reduce_op);
                case Type::STRING:
                    break;
                case Type::BINARY:
                    break;
                case Type::FIXED_SIZE_BINARY:
                    break;
                case Type::INTERVAL:
                    break;
                case Type::DECIMAL:
                    break;
                case Type::LIST:
                    break;
                case Type::EXTENSION:
                    break;
                case Type::FIXED_SIZE_LIST:
                    break;
                case Type::DURATION:
                    break;
                case Type::LARGE_STRING:
                    break;
                case Type::LARGE_BINARY:
                    break;
                case Type::MAX_ID:
                    break;
            }

            return {Code::NotImplemented, "allreduce not implemented for type"};
        }

        Status
        FmiAllgatherImpl::AllgatherBufferSize(const int32_t *send_data, int32_t num_buffers, int32_t *rcv_data) const {

            auto send_data_byte_size = num_buffers * sizeof(int32_t);
            auto send_void_ptr = const_cast<void *>(static_cast<const void *>(send_data));
            FMI::Comm::Data<void *> send_void_data(send_void_ptr, send_data_byte_size,
                                                   FMI::Comm::noop_deleter);
            auto recv_data_byte_size = comm_ptr_->getNumPeers() * num_buffers * sizeof(int32_t);
            auto recv_void_ptr = const_cast<void *>(static_cast<const void *>(rcv_data));
            FMI::Comm::Data<void *> recv_void_data(recv_void_ptr, recv_data_byte_size,
                                                   FMI::Comm::noop_deleter);
            comm_ptr_->allgather(send_void_data, recv_void_data, 0);
            return Status::OK();
        }

        Status FmiAllgatherImpl::IallgatherBufferData(int32_t buf_idx, const uint8_t *send_data, int32_t send_count,
                                                      uint8_t *recv_data, const std::vector<int32_t> &recv_count,
                                                      const std::vector<int32_t> &displacements) {
            auto send_data_byte_size = send_count * sizeof(uint8_t);
            auto send_void_ptr = const_cast<void *>(static_cast<const void *>(send_data));
            FMI::Comm::Data<void *> send_void_data(send_void_ptr, send_data_byte_size,
                                                   FMI::Comm::noop_deleter);

            std::size_t total_recv_size = 0;
            for (size_t i = 0; i < comm_ptr_->getNumPeers(); i++) {
                total_recv_size += recv_count[i];
            }

            auto recv_data_byte_size = total_recv_size * sizeof(uint8_t);
            auto recv_void_ptr = const_cast<void *>(static_cast<const void *>(recv_data));
            FMI::Comm::Data<void *> recv_void_data(recv_void_ptr, recv_data_byte_size,
                                                   FMI::Comm::noop_deleter);
            comm_ptr_->allgatherv(send_void_data, recv_void_data, 0, recv_count,
                                         displacements, mode_,
                                         [](FMI::Utils::NbxStatus status, const std::string &msg,
                                            FMI::Utils::fmiContext * ctx) {
                                             CYLON_UNUSED(ctx);
                                             if (status != FMI::Utils::SUCCESS) {
                                                 LOG(ERROR) << "FMI IallgatherBufferData status: "
                                                            << NbxStatusToString(status) << " msg: " << msg;
                                             }
                                         });

            return Status::OK();
        }

        Status FmiAllgatherImpl::WaitAll() {

            if (mode_ == FMI::Utils::NONBLOCKING) {
                while (comm_ptr_->communicator_event_progress(FMI::Utils::Operation::DEFAULT) ==
                       FMI::Utils::EventProcessStatus::PROCESSING) {}
            }

            return Status::OK();
        }

    }

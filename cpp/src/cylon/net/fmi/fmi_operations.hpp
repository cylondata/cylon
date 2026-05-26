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

#ifndef CYLON_FMI_OPERATIONS_HPP
#define CYLON_FMI_OPERATIONS_HPP

#include <cylon/net/comm_operations.hpp>
#include <cylon/thridparty/fmi/utils/Function.hpp>
#include "cylon/status.hpp"
#include "cylon/net/ops/base_ops.hpp"
#include <cylon/thridparty/fmi/Communicator.hpp>

namespace cylon::fmi {

        template<typename T>
        FMI::Utils::Function<T> get_function(net::ReduceOp reduce_op);

        class FmiTableAllgatherImpl : public net::TableAllgatherImpl {
        public:
            explicit FmiTableAllgatherImpl(const std::shared_ptr<FMI::Communicator> &comm_ptr,
                                           FMI::Utils::Mode mode)
                    : TableAllgatherImpl(), comm_ptr_(comm_ptr), mode_(mode) {}

            void Init(int num_buffers) override;

            Status AllgatherBufferSizes(const int32_t *send_data,
                                        int num_buffers,
                                        int32_t *rcv_data) const override;

            // gloo doesn't have non-blocking collectives. So, do blocking call here!
            Status IallgatherBufferData(int buf_idx,
                                        const uint8_t *send_data,
                                        int32_t send_count,
                                        uint8_t *recv_data,
                                        const std::vector<int32_t> &recv_count,
                                        const std::vector<int32_t> &displacements) override;

            Status WaitAll(int num_buffers) override;

        private:
            std::shared_ptr<FMI::Communicator> comm_ptr_;
            FMI::Utils::Mode mode_;
        };

        class FmiTableGatherImpl : public net::TableGatherImpl {
        public:
            explicit FmiTableGatherImpl(const std::shared_ptr<FMI::Communicator> &comm_ptr,
                                        FMI::Utils::Mode mode)
                    : comm_ptr_(comm_ptr), mode_(mode) {}

            void Init(int num_buffers) override;

            Status GatherBufferSizes(const int32_t *send_data,
                                     int num_buffers,
                                     int32_t *rcv_data,
                                     int gather_root) const override;

            Status IgatherBufferData(int buf_idx,
                                     const uint8_t *send_data,
                                     int32_t send_count,
                                     uint8_t *recv_data,
                                     const std::vector<int32_t> &recv_count,
                                     const std::vector<int32_t> &displacements,
                                     int gather_root) override;

            Status WaitAll(int num_buffers) override;

        private:
            std::shared_ptr<FMI::Communicator> comm_ptr_;
            FMI::Utils::Mode mode_;
        };

        class FmiTableBcastImpl : public net::TableBcastImpl {
        public:
            explicit FmiTableBcastImpl(const std::shared_ptr<FMI::Communicator> &comm_ptr,
                                       FMI::Utils::Mode mode)
                    : comm_ptr_(comm_ptr), mode_(mode) {}

            void Init(int32_t num_buffers) override;

            Status BcastBufferSizes(int32_t *buffer, int32_t count, int32_t bcast_root) const override;

            Status BcastBufferData(uint8_t *buf_data, int32_t send_count, int32_t bcast_root) const override;

            Status IbcastBufferData(int32_t buf_idx,
                                    uint8_t *buf_data,
                                    int32_t send_count,
                                    int32_t bcast_root) override;

            Status WaitAll(int32_t num_buffers) override;

        private:
            std::shared_ptr<FMI::Communicator> comm_ptr_;
            FMI::Utils::Mode mode_;
        };

        class FmiAllReduceImpl : public net::AllReduceImpl {
        public:
            explicit FmiAllReduceImpl(const std::shared_ptr<FMI::Communicator> & comm_ptr,
                                      FMI::Utils::Mode mode)
                    : comm_ptr_(comm_ptr), mode_(mode) {}

            Status AllReduceBuffer(const void *send_buf, void *rcv_buf, int count,
                                   const std::shared_ptr<DataType> &data_type,
                                   net::ReduceOp reduce_op) const override;

        private:
            const std::shared_ptr<FMI::Communicator> comm_ptr_;
            FMI::Utils::Mode mode_;
        };

        class FmiAllgatherImpl : public net::AllGatherImpl {
        public:
            explicit FmiAllgatherImpl(const std::shared_ptr<FMI::Communicator> & comm_ptr,
                                      FMI::Utils::Mode mode)
                    : comm_ptr_(comm_ptr), mode_(mode) {}

            Status AllgatherBufferSize(const int32_t *send_data,
                                       int32_t num_buffers,
                                       int32_t *rcv_data) const override;

            Status IallgatherBufferData(int32_t buf_idx,
                                        const uint8_t *send_data,
                                        int32_t send_count,
                                        uint8_t *recv_data,
                                        const std::vector<int32_t> &recv_count,
                                        const std::vector<int32_t> &displacements) override;

            Status WaitAll() override;

        private:
            std::shared_ptr<FMI::Communicator> comm_ptr_;
            FMI::Utils::Mode mode_;

        };
    }



#endif //CYLON_FMI_OPERATIONS_HPP

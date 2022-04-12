/*
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef CYLON_CPP_SRC_CYLON_NET_GLOO_GLOO_OPERATIONS_HPP_
#define CYLON_CPP_SRC_CYLON_NET_GLOO_GLOO_OPERATIONS_HPP_

#include <gloo/context.h>

#include "cylon/net/ops/base_ops.hpp"

namespace cylon {
namespace net {

class GlooTableAllgatherImpl : public TableAllgatherImpl {
 public:
  explicit GlooTableAllgatherImpl(const std::shared_ptr<gloo::Context> *ctx_ptr)
      : TableAllgatherImpl(), ctx_ptr_(ctx_ptr) {}

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
  const std::shared_ptr<gloo::Context> *ctx_ptr_;
};

class GlooTableGatherImpl : public TableGatherImpl {
 public:
  explicit GlooTableGatherImpl(const std::shared_ptr<gloo::Context> *ctx_ptr)
      : ctx_ptr_(ctx_ptr) {}

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
  const std::shared_ptr<gloo::Context> *ctx_ptr_;
};

class GlooTableBcastImpl : public TableBcastImpl {
 public:
  explicit GlooTableBcastImpl(const std::shared_ptr<gloo::Context> *ctx_ptr)
      : ctx_ptr_(ctx_ptr) {}

  void Init(int32_t num_buffers) override;

  Status BcastBufferSizes(int32_t *buffer, int32_t count, int32_t bcast_root) const override;

  Status BcastBufferData(uint8_t *buf_data, int32_t send_count, int32_t bcast_root) const override;

  Status IbcastBufferData(int32_t buf_idx,
                          uint8_t *buf_data,
                          int32_t send_count,
                          int32_t bcast_root) override;

  Status WaitAll(int32_t num_buffers) override;
 private:
  const std::shared_ptr<gloo::Context> *ctx_ptr_;
};

class GlooAllReduceImpl : public AllReduceImpl {
 public:
  explicit GlooAllReduceImpl(const std::shared_ptr<gloo::Context> *ctx_ptr)
      : ctx_ptr_(ctx_ptr) {}

  Status AllReduceBuffer(const void *send_buf, void *rcv_buf, int count,
                         const std::shared_ptr<DataType> &data_type,
                         ReduceOp reduce_op) const override;

 private:
  const std::shared_ptr<gloo::Context> *ctx_ptr_;
};

class GlooAllgatherImpl : public AllGatherImpl {
 public:
  explicit GlooAllgatherImpl(const std::shared_ptr<gloo::Context> *ctx_ptr)
      : ctx_ptr_(ctx_ptr) {}

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
  const std::shared_ptr<gloo::Context> *ctx_ptr_;
};

}
}
#endif //CYLON_CPP_SRC_CYLON_NET_GLOO_GLOO_OPERATIONS_HPP_

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

#include <gloo/allgather.h>
#include <gloo/allgatherv.h>
#include <gloo/allreduce.h>
#include <gloo/broadcast.h>
#include <gloo/gatherv.h>
#include <gloo/gather.h>
#include <gloo/math.h>

#include "gloo_operations.hpp"
#include "cylon/util/macros.hpp"

namespace cylon {
namespace net {

void GlooTableAllgatherImpl::Init(int num_buffers) {
  CYLON_UNUSED(num_buffers);
}

Status GlooTableAllgatherImpl::AllgatherBufferSizes(const int32_t *send_data,
                                                    int num_buffers,
                                                    int32_t *rcv_data) const {
  gloo::AllgatherOptions opts(*ctx_ptr_);
  opts.setInput(const_cast<int32_t *>(send_data), num_buffers);
  opts.setOutput(rcv_data, num_buffers * (*ctx_ptr_)->size);

  gloo::allgather(opts);

  return Status::OK();
}
Status GlooTableAllgatherImpl::IallgatherBufferData(int buf_idx,
                                                    const uint8_t *send_data,
                                                    int32_t send_count,
                                                    uint8_t *recv_data,
                                                    const std::vector<int32_t> &recv_count,
                                                    const std::vector<int32_t> &displacements) {
  CYLON_UNUSED(buf_idx);
  CYLON_UNUSED(displacements);

  gloo::AllgathervOptions opts(*ctx_ptr_);
  opts.setInput(const_cast<uint8_t *>(send_data), send_count);
  opts.setOutput(recv_data, std::vector<size_t>(recv_count.begin(), recv_count.end()));

  gloo::allgatherv(opts);
  return Status::OK();
}

Status GlooTableAllgatherImpl::WaitAll(int num_buffers) {
  CYLON_UNUSED(num_buffers);
  return Status::OK();
}

void GlooTableGatherImpl::Init(int num_buffers) {
  CYLON_UNUSED(num_buffers);
}

Status GlooTableGatherImpl::GatherBufferSizes(const int32_t *send_data,
                                              int num_buffers,
                                              int32_t *rcv_data,
                                              int gather_root) const {
  gloo::GatherOptions opts(*ctx_ptr_);
  opts.setInput(const_cast<int32_t *>(send_data), num_buffers);

  if (gather_root == (*ctx_ptr_)->rank) {
    opts.setOutput(rcv_data, num_buffers * (*ctx_ptr_)->size);
  } else {
    opts.setOutput(rcv_data, 0);
  }
  opts.setRoot(gather_root);

  gloo::gather(opts);
  return Status::OK();
}

Status GlooTableGatherImpl::IgatherBufferData(int buf_idx,
                                              const uint8_t *send_data,
                                              int32_t send_count,
                                              uint8_t *recv_data,
                                              const std::vector<int32_t> &recv_count,
                                              const std::vector<int32_t> &displacements,
                                              int gather_root) {
  CYLON_UNUSED(buf_idx);
  CYLON_UNUSED(displacements);

  gloo::GathervOptions opts(*ctx_ptr_);
  opts.setInput(const_cast<uint8_t *>(send_data), send_count);

  if (gather_root == (*ctx_ptr_)->rank) {
    opts.setOutput(recv_data, std::vector<size_t>(recv_count.begin(), recv_count.end()));
  } else {
    // Note: unlike MPI, gloo gets the send_count from elementsPerRank vector. So, it needs to be
    // sent explicitly!
    auto counts = std::vector<size_t>((*ctx_ptr_)->size, 0);
    counts[(*ctx_ptr_)->rank] = send_count;
    opts.setOutput<uint8_t>(recv_data, std::move(counts));
  }
  opts.setRoot(gather_root);

  gloo::gatherv(opts);

  return Status::OK();
}

Status GlooTableGatherImpl::WaitAll(int num_buffers) {
  CYLON_UNUSED(num_buffers);
  return Status::OK();
}

void GlooTableBcastImpl::Init(int32_t num_buffers) {
  CYLON_UNUSED(num_buffers);
}

Status GlooTableBcastImpl::BcastBufferSizes(int32_t *buffer,
                                            int32_t count,
                                            int32_t bcast_root) const {
  gloo::BroadcastOptions opts(*ctx_ptr_);

  opts.setRoot(bcast_root);
  if (bcast_root == (*ctx_ptr_)->rank) {
    opts.setInput(buffer, count);
  }

  opts.setOutput(buffer, count);

  gloo::broadcast(opts);
  return Status::OK();
}

Status GlooTableBcastImpl::BcastBufferData(uint8_t *buf_data,
                                           int32_t count,
                                           int32_t bcast_root) const {
  gloo::BroadcastOptions opts(*ctx_ptr_);

  opts.setRoot(bcast_root);
  if (bcast_root == (*ctx_ptr_)->rank) {
    opts.setInput(buf_data, count);
  }

  opts.setOutput(buf_data, count);

  gloo::broadcast(opts);
  return Status::OK();
}

Status GlooTableBcastImpl::IbcastBufferData(int32_t buf_idx,
                                            uint8_t *buf_data,
                                            int32_t send_count,
                                            int32_t bcast_root) {
  CYLON_UNUSED(buf_idx);
  return BcastBufferData(buf_data, send_count, bcast_root);
}

Status GlooTableBcastImpl::WaitAll(int32_t num_buffers) {
  CYLON_UNUSED(num_buffers);
  return Status::OK();
}

template<typename T>
gloo::AllreduceOptions::Func get_reduce_func(ReduceOp op) {
  void (*func)(void *, const void *, const void *, size_t);
  switch (op) {
    case SUM:func = &gloo::sum<T>;
      return func;
    case MIN:func = &gloo::min<T>;
      return func;
    case MAX:func = &gloo::max<T>;
      return func;
    case PROD:func = &gloo::product<T>;
      return func;
    case LAND:
    case LOR:
    case BAND:
    case BOR:return nullptr;
  }
  return nullptr;
}

template<typename T>
Status all_reduce_buffer(const std::shared_ptr<gloo::Context> &ctx,
                         const void *send_buf,
                         void *rcv_buf,
                         int count,
                         ReduceOp reduce_op) {
  gloo::AllreduceOptions opts(ctx);
  opts.setReduceFunction(get_reduce_func<T>(reduce_op));

  opts.template setInput<T>(const_cast<T *>((const T *) send_buf), count);
  opts.template setOutput<T>((T *) rcv_buf, count);

  gloo::allreduce(opts);
  return Status::OK();
}

Status GlooAllReduceImpl::AllReduceBuffer(const void *send_buf, void *rcv_buf, int count,
                                          const std::shared_ptr<DataType> &data_type,
                                          ReduceOp reduce_op) const {
  switch (data_type->getType()) {
    case Type::BOOL:break;
    case Type::UINT8:
      return all_reduce_buffer<uint8_t>(*ctx_ptr_,
                                        send_buf,
                                        rcv_buf,
                                        count,
                                        reduce_op);
    case Type::INT8:
      return all_reduce_buffer<int8_t>(*ctx_ptr_,
                                       send_buf,
                                       rcv_buf,
                                       count,
                                       reduce_op);
    case Type::UINT16:
      return all_reduce_buffer<uint16_t>(*ctx_ptr_,
                                         send_buf,
                                         rcv_buf,
                                         count,
                                         reduce_op);
    case Type::INT16:
      return all_reduce_buffer<int16_t>(*ctx_ptr_,
                                        send_buf,
                                        rcv_buf,
                                        count,
                                        reduce_op);
    case Type::UINT32:
      return all_reduce_buffer<uint32_t>(*ctx_ptr_,
                                         send_buf,
                                         rcv_buf,
                                         count,
                                         reduce_op);
    case Type::INT32:
      return all_reduce_buffer<int32_t>(*ctx_ptr_,
                                        send_buf,
                                        rcv_buf,
                                        count,
                                        reduce_op);
    case Type::UINT64:
      return all_reduce_buffer<uint64_t>(*ctx_ptr_,
                                         send_buf,
                                         rcv_buf,
                                         count,
                                         reduce_op);
    case Type::INT64:
      return all_reduce_buffer<int64_t>(*ctx_ptr_,
                                        send_buf,
                                        rcv_buf,
                                        count,
                                        reduce_op);
    case Type::HALF_FLOAT:break;
    case Type::FLOAT:
      return all_reduce_buffer<float>(*ctx_ptr_,
                                      send_buf,
                                      rcv_buf,
                                      count,
                                      reduce_op);
    case Type::DOUBLE:
      return all_reduce_buffer<double>(*ctx_ptr_,
                                       send_buf,
                                       rcv_buf,
                                       count,
                                       reduce_op);
    case Type::DATE32:
    case Type::TIME32:
      return all_reduce_buffer<uint32_t>(*ctx_ptr_,
                                         send_buf,
                                         rcv_buf,
                                         count,
                                         reduce_op);
    case Type::DATE64:
    case Type::TIMESTAMP:
    case Type::TIME64:
      return all_reduce_buffer<uint64_t>(*ctx_ptr_,
                                         send_buf,
                                         rcv_buf,
                                         count,
                                         reduce_op);
    case Type::STRING:break;
    case Type::BINARY:break;
    case Type::FIXED_SIZE_BINARY:break;
    case Type::INTERVAL:break;
    case Type::DECIMAL:break;
    case Type::LIST:break;
    case Type::EXTENSION:break;
    case Type::FIXED_SIZE_LIST:break;
    case Type::DURATION:break;
    case Type::LARGE_STRING:break;
    case Type::LARGE_BINARY:break;
    case Type::MAX_ID:break;
  }
  return {Code::NotImplemented, "allreduce not implemented for type"};
}

Status GlooAllgatherImpl::AllgatherBufferSize(const int32_t *send_data,
                                              int32_t num_buffers,
                                              int32_t *rcv_data) const {
  gloo::AllgatherOptions opts(*ctx_ptr_);
  opts.setInput(const_cast<int32_t *>(send_data), num_buffers);
  opts.setOutput(rcv_data, num_buffers * (*ctx_ptr_)->size);

  gloo::allgather(opts);

  return Status::OK();
}

Status GlooAllgatherImpl::IallgatherBufferData(int32_t buf_idx,
                                               const uint8_t *send_data,
                                               int32_t send_count,
                                               uint8_t *recv_data,
                                               const std::vector<int32_t> &recv_count,
                                               const std::vector<int32_t> &displacements) {
  CYLON_UNUSED(buf_idx);
  CYLON_UNUSED(displacements);

  gloo::AllgathervOptions opts(*ctx_ptr_);
  opts.setInput(const_cast<uint8_t *>(send_data), send_count);
  opts.setOutput(recv_data, std::vector<size_t>(recv_count.begin(), recv_count.end()));

  gloo::allgatherv(opts);
  return Status::OK();
}

Status GlooAllgatherImpl::WaitAll() {
  return Status::OK();
}
}
}
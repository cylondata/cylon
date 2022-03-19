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
#include <gloo/broadcast.h>
#include <gloo/gatherv.h>
#include <gloo/gather.h>

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
}
}
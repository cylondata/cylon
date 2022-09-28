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

#include <ucc/api/ucc.h>
#include <ucc/api/ucc_def.h>

#include "cylon/util/macros.hpp"
#include "cylon/net/comm_operations.hpp"
#include "cylon/net/ops/base_ops.hpp"
#include "cylon/net/comm_operations.hpp"

namespace cylon {
namespace ucc {

class UccTableAllgatherImpl : public net::TableAllgatherImpl {
 public:
  UccTableAllgatherImpl(ucc_team_h ucc_team, ucc_context_h ucc_context, int world_size);
  ~UccTableAllgatherImpl() override = default;

  void Init(int num_buffers) override;

  Status AllgatherBufferSizes(const int32_t *send_data, int num_buffers,
                              int32_t *rcv_data) const override;

  Status IallgatherBufferData(
      int buf_idx, const uint8_t *send_data, int32_t send_count,
      uint8_t *recv_data, const std::vector<int32_t> &recv_count,
      const std::vector<int32_t> &displacements) override;

  Status WaitAll(int num_buffers) override;

 private:
  ucc_team_h ucc_team_;
  ucc_context_h ucc_context_;
  std::vector<ucc_coll_req_h> requests_;
  std::vector<ucc_coll_args_t> args_;
  int world_size;
};

class UccTableGatherImpl : public net::TableGatherImpl {
 public:
  UccTableGatherImpl(ucc_team_h ucc_team, ucc_context_h ucc_context, int rank,
                     int world_size);

  ~UccTableGatherImpl() override;

  void Init(int32_t num_buffers) override;

  Status GatherBufferSizes(const int32_t *send_data, int32_t num_buffers,
                           int32_t *rcv_data,
                           int32_t gather_root) const override;

  Status IgatherBufferData(int32_t buf_idx, const uint8_t *send_data,
                           int32_t send_count, uint8_t *recv_data,
                           const std::vector<int32_t> &recv_count,
                           const std::vector<int32_t> &displacements,
                           int32_t gather_root) override;

  Status WaitAll(int32_t num_buffers) override;

 private:
  // the following three are to mimic gather using allgather
  std::vector<std::vector<int>>* displacements_;
  std::vector<std::vector<int>>* all_recv_counts_;
  uint8_t* recv_data_placeholder;

  std::vector<ucc_coll_req_h> requests_;
  std::vector<ucc_coll_args_t> args_;
  ucc_team_h ucc_team_;
  ucc_context_h ucc_context_;
  int world_size;
  int rank;
};

class UccAllReduceImpl : public net::AllReduceImpl {
public:
  UccAllReduceImpl(ucc_team_h ucc_team, ucc_context_h ucc_context);
  ~UccAllReduceImpl() override = default;

  Status AllReduceBuffer(const void *send_buf, void *rcv_buf, int count,
                         const std::shared_ptr<DataType> &data_type,
                         net::ReduceOp reduce_op) const override;

private:
  ucc_team_h ucc_team_;
  ucc_context_h ucc_context_;
};

class UccTableBcastImpl : public net::TableBcastImpl {
 public:
  UccTableBcastImpl(ucc_team_h ucc_team, ucc_context_h ucc_context);
  ~UccTableBcastImpl() override = default;
  void Init(int32_t num_buffers) override;
  Status BcastBufferSizes(int32_t *buffer, int32_t count,
                          int32_t bcast_root) const override;
  Status BcastBufferData(uint8_t *buf_data, int32_t send_count,
                         int32_t bcast_root) const override;
  Status IbcastBufferData(int32_t buf_idx, uint8_t *buf_data,
                          int32_t send_count, int32_t bcast_root) override;
  Status WaitAll(int32_t num_buffers) override;
 private:
  ucc_team_h ucc_team_;
  ucc_context_h ucc_context_;
  std::vector<ucc_coll_req_h> reqs;
  std::vector<ucc_coll_args_t> args;
};

class UccAllGatherImpl : public net::AllGatherImpl {
 public:
  UccAllGatherImpl(ucc_team_h ucc_team, ucc_context_h ucc_context, int world_size);
  ~UccAllGatherImpl() override = default;
  Status AllgatherBufferSize(const int32_t *send_data, int32_t num_buffers,
                             int32_t *rcv_data) const override;
  Status IallgatherBufferData(int32_t buf_idx, const uint8_t *send_data,
                              int32_t send_count, uint8_t *recv_data,
                              const std::vector<int32_t> &recv_count,
                              const std::vector<int32_t> &displacements) override;
  Status WaitAll() override;
 private:
  ucc_team_h ucc_team_;
  ucc_context_h ucc_context_;
  std::vector<ucc_coll_req_h> requests_;
  std::vector<ucc_coll_args_t> args_;
  int world_size;
};

}  // namespace ucc
}  // namespace cylon
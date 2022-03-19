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

#ifndef CYLON_CPP_SRC_CYLON_NET_OPS_BASE_OPS_HPP_
#define CYLON_CPP_SRC_CYLON_NET_OPS_BASE_OPS_HPP_

#include "cylon/net/serialize.hpp"
#include "cylon/net/buffer.hpp"
#include "cylon/data_types.hpp"
#include "cylon/net/comm_operations.hpp"
#include "cylon/column.hpp"

namespace cylon {
class CylonContext;
class Table;
class Column;
class Scalar;

namespace net {

class TableAllgatherImpl {
 public:
  virtual ~TableAllgatherImpl() = default;

  virtual void Init(int32_t num_buffers) = 0;

  virtual Status AllgatherBufferSizes(const int32_t *send_data,
                                      int32_t num_buffers,
                                      int32_t *rcv_data) const = 0;

  virtual Status IallgatherBufferData(int32_t buf_idx,
                                      const uint8_t *send_data,
                                      int32_t send_count,
                                      uint8_t *recv_data,
                                      const std::vector<int32_t> &recv_count,
                                      const std::vector<int32_t> &displacements) = 0;

  virtual Status WaitAll(int32_t num_buffers) = 0;

  Status Execute(const std::shared_ptr<TableSerializer> &serializer,
                 const std::shared_ptr<cylon::Allocator> &allocator,
                 int32_t world_size,
                 std::vector<int32_t> *all_buffer_sizes,
                 std::vector<std::shared_ptr<Buffer>> *received_buffers,
                 std::vector<std::vector<int32_t>> *displacements);
};
Status DoTableAllgather(TableAllgatherImpl &impl,
                        const std::shared_ptr<Table> &table,
                        std::vector<std::shared_ptr<Table>> *out);

class TableGatherImpl {
 public:
  virtual ~TableGatherImpl() = default;

  virtual void Init(int32_t num_buffers) = 0;

  virtual Status GatherBufferSizes(const int32_t *send_data,
                                   int32_t num_buffers,
                                   int32_t *rcv_data,
                                   int32_t gather_root) const = 0;

  virtual Status IgatherBufferData(int32_t buf_idx,
                                   const uint8_t *send_data,
                                   int32_t send_count,
                                   uint8_t *recv_data,
                                   const std::vector<int32_t> &recv_count,
                                   const std::vector<int32_t> &displacements,
                                   int32_t gather_root) = 0;

  virtual Status WaitAll(int32_t num_buffers) = 0;

  Status Execute(const std::shared_ptr<TableSerializer> &serializer,
                 const std::shared_ptr<Allocator> &allocator,
                 int32_t rank,
                 int32_t world_size,
                 int32_t gather_root,
                 bool gather_from_root,
                 std::vector<int32_t> *all_buffer_sizes,
                 std::vector<std::shared_ptr<Buffer>> *received_buffers,
                 std::vector<std::vector<int32_t>> *displacements);
};
Status DoTableGather(TableGatherImpl &impl,
                     const std::shared_ptr<Table> &table,
                     int32_t gather_root,
                     bool gather_from_root,
                     std::vector<std::shared_ptr<Table>> *out);

class TableBcastImpl {
 public:
  virtual ~TableBcastImpl() = default;

  virtual void Init(int32_t num_buffers) = 0;

  virtual Status BcastBufferSizes(int32_t *buffer,
                                  int32_t count,
                                  int32_t bcast_root) const = 0;

  virtual Status BcastBufferData(uint8_t *buf_data,
                                 int32_t send_count,
                                 int32_t bcast_root) const = 0;

  virtual Status IbcastBufferData(int32_t buf_idx,
                                  uint8_t *buf_data,
                                  int32_t send_count,
                                  int32_t bcast_root) = 0;

  virtual Status WaitAll(int32_t num_buffers) = 0;

  Status Execute(const std::shared_ptr<TableSerializer> &serializer,
                 const std::shared_ptr<Allocator> &allocator,
                 int32_t rank,
                 int32_t bcast_root,
                 std::vector<std::shared_ptr<Buffer>> *received_buffers,
                 std::vector<int32_t> *data_types);
};

Status DoTableBcast(TableBcastImpl &impl, std::shared_ptr<Table> *table, int bcast_root,
                    const std::shared_ptr<CylonContext> &ctx);

class AllReduceImpl {
 public:
  virtual ~AllReduceImpl() = default;

  virtual Status AllReduceBuffer(const void *send_buf,
                                 void *rcv_buf,
                                 int count,
                                 const std::shared_ptr<DataType> &data_type,
                                 ReduceOp reduce_op) const = 0;

  Status Execute(const std::shared_ptr<Column> &values, net::ReduceOp reduce_op,
                 std::shared_ptr<Column> *output, MemoryPool *pool = nullptr) const;

  Status Execute(const std::shared_ptr<Scalar> &value, net::ReduceOp reduce_op,
                 std::shared_ptr<Scalar> *output, MemoryPool *pool = nullptr) const;
};

}
}

#endif //CYLON_CPP_SRC_CYLON_NET_OPS_BASE_OPS_HPP_

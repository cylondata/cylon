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

#ifndef CYLON_SRC_CYLON_COMM_COMMUNICATOR_H_
#define CYLON_SRC_CYLON_COMM_COMMUNICATOR_H_

#include "cylon/net/comm_config.hpp"
#include "cylon/net/channel.hpp"
#include "cylon/net/comm_operations.hpp"

namespace cylon {
class CylonContext;
class Table;
class Column;

namespace net {

class Communicator {
 protected:
  int rank = -1;
  int world_size = -1;
 public:
  virtual Status Init(const std::shared_ptr<CommConfig> &config) = 0;
  virtual std::unique_ptr<Channel> CreateChannel() const = 0;
  virtual int GetRank() const = 0;
  virtual int GetWorldSize() const = 0;
  virtual void Finalize() = 0;
  virtual void Barrier() = 0;
  virtual CommType GetCommType() const = 0;

  virtual ~Communicator() = default;

  virtual Status AllGather(const std::shared_ptr<Table> &table,
                           std::vector<std::shared_ptr<Table>> *out) const = 0;

  virtual Status Gather(const std::shared_ptr<Table> &table,
                        int gather_root,
                        bool gather_from_root,
                        std::vector<std::shared_ptr<Table>> *out) const = 0;

  virtual Status Bcast(const std::shared_ptr<CylonContext> &ctx,
                       std::shared_ptr<Table> *table,
                       int bcast_root) const = 0;

  /**
   * Allreduce values at every index on `values`.
   * @param ctx
   * @param values
   * @param reduce_op
   * @param output
   * @param skip_nulls if `true`,
   * @return
   */
  virtual Status AllReduce(const std::shared_ptr<CylonContext> &ctx,
                           const std::shared_ptr<Column> &values,
                           net::ReduceOp reduce_op,
                           std::shared_ptr<Column> *output) const = 0;
};
}
}

#endif //CYLON_SRC_CYLON_COMM_COMMUNICATOR_H_

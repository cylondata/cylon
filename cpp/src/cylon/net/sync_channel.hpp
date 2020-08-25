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

#ifndef CYLON_CPP_SRC_CYLON_NET_SYNC_CHANNEL_HPP_
#define CYLON_CPP_SRC_CYLON_NET_SYNC_CHANNEL_HPP_
#include <mpi.h>
#include <data_types.hpp>

namespace cylon {

/**
 * Cylon reduction operations
 */
enum ReduceOp {
  SUM,
  MIN,
  MAX
};

/**
 * Channel for syncronous operations
 */
class SyncChannel {
 public:
  explicit SyncChannel() = default;

  /**
   * All reduce
   * @param send_buf
   * @param rcv_buf
   * @param count
   * @param data_type cylon::Datatype
   * @param reduce_op
   * @return
   */
  virtual int AllReduce(void *send_buf, void *rcv_buf, int count, const std::shared_ptr<DataType> &data_type,
                        ReduceOp reduce_op) = 0;

};

}
#endif //CYLON_CPP_SRC_CYLON_NET_SYNC_CHANNEL_HPP_

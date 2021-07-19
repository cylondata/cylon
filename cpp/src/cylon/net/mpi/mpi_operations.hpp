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

#ifndef CYLON_CPP_SRC_CYLON_NET_MPI_MPI_OPERATIONS_HPP_
#define CYLON_CPP_SRC_CYLON_NET_MPI_MPI_OPERATIONS_HPP_

#include <mpi.h>
#include <cylon/net/comm_operations.hpp>

namespace cylon {
namespace mpi {

MPI_Op GetMPIOp(cylon::net::ReduceOp reduce_op);

MPI_Datatype GetMPIDataType(const std::shared_ptr<DataType> &data_type);

cylon::Status AllReduce(const void *send_buf,
                        void *rcv_buf,
                        int count,
                        const std::shared_ptr<DataType> &data_type,
                        cylon::net::ReduceOp reduce_op);

}
}
#endif //CYLON_CPP_SRC_CYLON_NET_MPI_MPI_OPERATIONS_HPP_

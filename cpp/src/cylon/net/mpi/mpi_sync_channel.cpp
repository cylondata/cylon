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

#include "mpi_sync_channel.hpp"

MPI_Op cylon::GetMPIOp(cylon::ReduceOp reduce_op) {
  switch (reduce_op) {
    case cylon::SUM: return MPI_SUM;
    case cylon::MIN: return MPI_MIN;
    case cylon::MAX: return MPI_MAX;
//    case cylon::PROD: return MPI_PROD;
    default: return nullptr;
  }
}

MPI_Datatype cylon::GetMPIDataType(const std::shared_ptr<DataType> &data_type) {
  switch (data_type->getType()) {
    case Type::BOOL:return MPI_CXX_BOOL;
    case Type::UINT8:return MPI_UINT8_T;
    case Type::INT8:return MPI_INT8_T;
    case Type::UINT16:return MPI_UINT16_T;
    case Type::INT16:return MPI_INT16_T;
    case Type::UINT32:return MPI_UINT32_T;
    case Type::INT32:return MPI_INT32_T;
    case Type::UINT64:return MPI_UINT32_T;
    case Type::INT64:return MPI_INT64_T;
    case Type::FLOAT:return MPI_FLOAT;
    case Type::DOUBLE:return MPI_DOUBLE;
    case Type::STRING:
    case Type::BINARY:
    case Type::FIXED_SIZE_BINARY: return MPI_BYTE;
      //todo: MPI does not support 16byte floats. We'll have to use a custom datatype for this later.
    case Type::HALF_FLOAT:
    case Type::DATE32:
    case Type::DATE64:
    case Type::TIMESTAMP:
    case Type::TIME32:
    case Type::TIME64:
    case Type::DECIMAL:
    case Type::DURATION:
    case Type::INTERVAL:
    case Type::LIST:
    case Type::FIXED_SIZE_LIST:
    case Type::EXTENSION:break;
  }
  return nullptr;
}

int cylon::MPISyncChannel::AllReduce(void *send_buf,
                                     void *rcv_buf,
                                     int count,
                                     const std::shared_ptr<DataType> &data_type,
                                     cylon::ReduceOp reduce_op) {

  MPI_Datatype mpi_data_type = cylon::GetMPIDataType(data_type);
  MPI_Op mpi_op = cylon::GetMPIOp(reduce_op);

  if (mpi_data_type == nullptr || mpi_op == nullptr) return 1;

  return MPI_Allreduce(send_buf, rcv_buf, count, mpi_data_type, mpi_op, MPI_COMM_WORLD);
}


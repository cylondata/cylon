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

#include <status.hpp>
#include "mpi_operations.hpp"

MPI_Op cylon::mpi::GetMPIOp(cylon::net::ReduceOp reduce_op) {
  switch (reduce_op) {
    case cylon::net::SUM: return MPI_SUM;
    case cylon::net::MIN: return MPI_MIN;
    case cylon::net::MAX: return MPI_MAX;
//    case cylon::PROD: return MPI_PROD;
    default: return nullptr;
  }
}

MPI_Datatype cylon::mpi::GetMPIDataType(const std::shared_ptr<DataType> &data_type) {
  switch (data_type->getType()) {
    case Type::BOOL:return MPI_CXX_BOOL;
    case Type::UINT8:return MPI_UINT8_T;
    case Type::INT8:return MPI_INT8_T;
    case Type::UINT16:return MPI_UINT16_T;
    case Type::INT16:return MPI_INT16_T;
    case Type::UINT32:return MPI_UINT32_T;
    case Type::INT32:return MPI_INT32_T;
    case Type::UINT64:return MPI_UINT64_T;
    case Type::INT64:return MPI_INT64_T;
    case Type::FLOAT:return MPI_FLOAT;
    case Type::DOUBLE:return MPI_DOUBLE;
    case Type::STRING:
    case Type::BINARY:
    case Type::FIXED_SIZE_BINARY: return MPI_BYTE;
      //todo: MPI does not support 16byte floats. We'll have to use a custom datatype for this later.
    case Type::HALF_FLOAT:
    case Type::DATE32:return MPI_UINT32_T;
    case Type::DATE64:return MPI_UINT64_T;
    case Type::TIMESTAMP:return MPI_UINT64_T;
    case Type::TIME32:return MPI_UINT32_T;
    case Type::TIME64:return MPI_UINT64_T;
    case Type::DECIMAL:
    case Type::DURATION:
    case Type::INTERVAL:
    case Type::LIST:
    case Type::FIXED_SIZE_LIST:
    case Type::EXTENSION:break;
  }
  return nullptr;
}

cylon::Status cylon::mpi::AllReduce(const void *send_buf,
                                    void *rcv_buf,
                                    const int count,
                                    const std::shared_ptr<cylon::DataType> &data_type,
                                    const cylon::net::ReduceOp reduce_op) {
  MPI_Datatype mpi_data_type = cylon::mpi::GetMPIDataType(data_type);
  MPI_Op mpi_op = cylon::mpi::GetMPIOp(reduce_op);

  if (mpi_data_type == nullptr || mpi_op == nullptr) {
    return cylon::Status(cylon::Code::NotImplemented, "Unknown data type or operation for MPI");
  }

  if (MPI_Allreduce(send_buf, rcv_buf, count, mpi_data_type, mpi_op, MPI_COMM_WORLD) == MPI_SUCCESS) {
    return cylon::Status::OK();
  } else {
    return cylon::Status(cylon::Code::ExecutionError, "MPI operation failed!");
  }
}


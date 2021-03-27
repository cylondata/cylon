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

#ifndef CYLON_CPP_SRC_CYLON_COMPUTE_AGGREGATE_UTILS_HPP_
#define CYLON_CPP_SRC_CYLON_COMPUTE_AGGREGATE_UTILS_HPP_

#include "../net/mpi/mpi_operations.hpp"

namespace cylon {
namespace compute {

/**
 * All reduce for numeric types
 * @tparam NUM_ARROW_T arrow numeric type
 * @param comm_type
 * @param send sending container
 * @param output output result
 * @param data_type
 * @param reduce_op
 * @return
 */
template<typename NUM_ARROW_T,
    typename = typename std::enable_if<
        arrow::is_number_type<NUM_ARROW_T>::value
            | arrow::is_boolean_type<NUM_ARROW_T>::value>::type>
cylon::Status AllReduce(cylon::net::CommType comm_type,
                        const arrow::Datum &send,
                        std::shared_ptr<Result> &output,
                        const std::shared_ptr<DataType> &data_type,
                        cylon::net::ReduceOp reduce_op) {
  using NUM_ARROW_SCALAR_T = typename arrow::TypeTraits<NUM_ARROW_T>::ScalarType;

  const std::shared_ptr<NUM_ARROW_SCALAR_T>
      &send_scalar = std::static_pointer_cast<NUM_ARROW_SCALAR_T>(send.scalar());
  std::shared_ptr<NUM_ARROW_SCALAR_T> recv_scalar = std::make_shared<NUM_ARROW_SCALAR_T>();

  switch (comm_type) {
    case net::LOCAL: {
      output = std::make_shared<Result>(send);
      return cylon::Status::OK();
    }
    case cylon::net::CommType::MPI: {
      cylon::Status status = cylon::mpi::AllReduce(&(send_scalar->value),
                                                   &(recv_scalar->value),
                                                   send.length(),
                                                   data_type,
                                                   reduce_op);
      // build the output datum
      if (status.is_ok()) {
        arrow::Datum global_result(recv_scalar);
        output = std::make_shared<Result>(global_result);
      }

      return status;
    }
    default: return cylon::Status(cylon::Code::NotImplemented, "Mode is not supported!");
  }
}

template<typename NUM_ARROW_T,
    typename = typename std::enable_if<
        arrow::is_number_type<NUM_ARROW_T>::value
            | arrow::is_boolean_type<NUM_ARROW_T>::value>::type>
cylon::Status AllReduce(cylon::net::CommType comm_type,
                        const arrow::Datum &send,
                        std::shared_ptr<Result> &output,
                        const std::shared_ptr<DataType> &data_type,
                        const std::vector<cylon::net::ReduceOp> &reduce_ops) {
  using NUM_ARROW_SCALAR_T = typename arrow::TypeTraits<NUM_ARROW_T>::ScalarType;

  const std::shared_ptr<arrow::StructScalar>
      &send_struct_scalar = std::static_pointer_cast<arrow::StructScalar>(send.scalar());

  if (reduce_ops.size() != send_struct_scalar->value.size()) {
    return cylon::Status(cylon::Code::ExecutionError, "reduce ops size != scalar vector size");
  }

  arrow::ScalarVector rcv_scalar_vector;
  rcv_scalar_vector.reserve(reduce_ops.size());

  switch (comm_type) {
    case net::LOCAL: {
      output = std::make_shared<Result>(send);
      return cylon::Status::OK();
    }
    case cylon::net::CommType::MPI: {
      for (size_t i = 0; i < reduce_ops.size(); i++) {
        const std::shared_ptr<NUM_ARROW_SCALAR_T>
            &send_scalar =
            std::static_pointer_cast<NUM_ARROW_SCALAR_T>(send_struct_scalar->value[i]);
        std::shared_ptr<NUM_ARROW_SCALAR_T> rcv_scalar = std::make_shared<NUM_ARROW_SCALAR_T>();

        RETURN_CYLON_STATUS_IF_FAILED(cylon::mpi::AllReduce(send_scalar->data(),
                                                            rcv_scalar->mutable_data(),
                                                            1, data_type, reduce_ops[i]));
        rcv_scalar_vector.push_back(rcv_scalar);
      }
      auto rcv_struct_scalar = std::make_shared<arrow::StructScalar>(rcv_scalar_vector, send_struct_scalar->type);
      // build the output datum
      arrow::Datum global_result(rcv_struct_scalar);
      output = std::make_shared<Result>(global_result);

      return Status::OK();
    }
    default: return cylon::Status(cylon::Code::NotImplemented, "Mode is not supported!");
  }
}

template<typename RED_OPS=cylon::net::ReduceOp>
cylon::Status DoAllReduce(std::shared_ptr<cylon::CylonContext> &ctx,
                          const arrow::Datum &snd,
                          std::shared_ptr<Result> &rcv,
                          const std::shared_ptr<DataType> &dtype,
                          const RED_OPS &red_op) {
  auto comm_type = ctx->GetCommType();
  switch (dtype->getType()) {
    case Type::BOOL:return AllReduce<arrow::BooleanType>(comm_type, snd, rcv, dtype, red_op);
    case Type::UINT8:return AllReduce<arrow::UInt8Type>(comm_type, snd, rcv, dtype, red_op);
    case Type::INT8:return AllReduce<arrow::Int8Type>(comm_type, snd, rcv, dtype, red_op);
    case Type::UINT16:return AllReduce<arrow::UInt16Type>(comm_type, snd, rcv, dtype, red_op);
    case Type::INT16:return AllReduce<arrow::Int16Type>(comm_type, snd, rcv, dtype, red_op);
    case Type::UINT32:return AllReduce<arrow::UInt32Type>(comm_type, snd, rcv, dtype, red_op);
    case Type::INT32:return AllReduce<arrow::Int32Type>(comm_type, snd, rcv, dtype, red_op);
    case Type::UINT64:return AllReduce<arrow::UInt64Type>(comm_type, snd, rcv, dtype, red_op);
    case Type::INT64:return AllReduce<arrow::Int64Type>(comm_type, snd, rcv, dtype, red_op);
    case Type::FLOAT:return AllReduce<arrow::FloatType>(comm_type, snd, rcv, dtype, red_op);
    case Type::DOUBLE:return AllReduce<arrow::DoubleType>(comm_type, snd, rcv, dtype, red_op);
    default: return cylon::Status(cylon::Code::Invalid, "data type not supported!");
  }
}

}
}
#endif //CYLON_CPP_SRC_CYLON_COMPUTE_AGGREGATE_UTILS_HPP_

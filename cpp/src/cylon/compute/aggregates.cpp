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

#include <glog/logging.h>
#include <arrow/compute/kernels/minmax.h> // minmax kernel is not included in the arrrow/compute/api.h

#include <status.hpp>
#include <table.hpp>
#include <ctx/arrow_memory_pool_utils.hpp>
#include <net/mpi/mpi_operations.hpp>

#include "compute/aggregates.hpp"

namespace cylon {
namespace compute {

//template<typename NUM_ARROW_T>
template<typename NUM_ARROW_T,
    typename = typename std::enable_if<
        arrow::is_number_type<NUM_ARROW_T>::value | arrow::is_boolean_type<NUM_ARROW_T>::value>::type>
cylon::Status AllReduce(cylon::net::CommType comm_type,
                        const arrow::compute::Datum &send,
                        std::shared_ptr<Result> *output,
                        const std::shared_ptr<DataType> &data_type,
                        cylon::ReduceOp reduce_op) {
  using NUM_ARROW_SCALAR_T = typename arrow::TypeTraits<NUM_ARROW_T>::ScalarType;

  const shared_ptr<NUM_ARROW_SCALAR_T> &send_scalar = std::static_pointer_cast<NUM_ARROW_SCALAR_T>(send.scalar());
  std::shared_ptr<NUM_ARROW_SCALAR_T> recv_scalar = std::make_shared<NUM_ARROW_SCALAR_T>();

  switch (comm_type) {
    case net::LOCAL: {
      *output = std::make_shared<Result>(send);
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
        arrow::compute::Datum global_result(recv_scalar);
        *output = std::make_shared<Result>(global_result);
      }

      return status;
    }
    case net::TCP: // fall through
    case net::UCX: // fall through
    default: return cylon::Status(cylon::Code::NotImplemented, "Mode is not supported!");
  }
}

cylon::Status DoAllReduce(cylon::CylonContext *ctx,
                          const arrow::compute::Datum &send,
                          std::shared_ptr<Result> *receive,
                          const shared_ptr<DataType> &data_type,
                          cylon::ReduceOp reduce_op) {
  auto comm_type = ctx->GetCommType();
  switch (data_type->getType()) {
    case Type::BOOL:
      return cylon::compute::AllReduce<arrow::BooleanType>(comm_type,
                                                           send,
                                                           receive,
                                                           data_type,
                                                           reduce_op);
    case Type::UINT8:
      return cylon::compute::AllReduce<arrow::UInt8Type>(comm_type,
                                                         send,
                                                         receive,
                                                         data_type,
                                                         reduce_op);
    case Type::INT8:
      return cylon::compute::AllReduce<arrow::Int8Type>(comm_type,
                                                        send,
                                                        receive,
                                                        data_type,
                                                        reduce_op);
    case Type::UINT16:
      return cylon::compute::AllReduce<arrow::UInt16Type>(comm_type,
                                                          send,
                                                          receive,
                                                          data_type,
                                                          reduce_op);
    case Type::INT16:
      return cylon::compute::AllReduce<arrow::Int16Type>(comm_type,
                                                         send,
                                                         receive,
                                                         data_type,
                                                         reduce_op);
    case Type::UINT32:
      return cylon::compute::AllReduce<arrow::UInt32Type>(comm_type,
                                                          send,
                                                          receive,
                                                          data_type,
                                                          reduce_op);
    case Type::INT32:
      return cylon::compute::AllReduce<arrow::Int32Type>(comm_type,
                                                         send,
                                                         receive,
                                                         data_type,
                                                         reduce_op);
    case Type::UINT64:
      return cylon::compute::AllReduce<arrow::UInt64Type>(comm_type,
                                                          send,
                                                          receive,
                                                          data_type,
                                                          reduce_op);
    case Type::INT64:
      return cylon::compute::AllReduce<arrow::Int64Type>(comm_type,
                                                         send,
                                                         receive,
                                                         data_type,
                                                         reduce_op);
    case Type::FLOAT:
      return cylon::compute::AllReduce<arrow::FloatType>(comm_type,
                                                         send,
                                                         receive,
                                                         data_type,
                                                         reduce_op);
    case Type::DOUBLE:
      return cylon::compute::AllReduce<arrow::DoubleType>(comm_type,
                                                          send,
                                                          receive,
                                                          data_type,
                                                          reduce_op);
    case Type::HALF_FLOAT:
    case Type::STRING:
    case Type::BINARY:
    case Type::FIXED_SIZE_BINARY:
    case Type::DATE32:
    case Type::DATE64:
    case Type::TIMESTAMP:
    case Type::TIME32:
    case Type::TIME64:
    case Type::INTERVAL:
    case Type::DECIMAL:
    case Type::LIST:
    case Type::EXTENSION:
    case Type::FIXED_SIZE_LIST:
    case Type::DURATION:
    default: return cylon::Status(cylon::Code::Invalid, "data type not supported!");
  }
}


cylon::Status Sum(cylon::CylonContext *ctx,
                  const std::shared_ptr<cylon::Table> &table,
                  int32_t col_idx,
                  std::shared_ptr<Result> *output) {
  const shared_ptr<Column> &col = table->GetColumn(col_idx); // cylon column object
  const shared_ptr<DataType> &data_type = col->GetDataType();
  const arrow::compute::Datum input(col->GetColumnData()); // input datum

  // do local operation
  arrow::compute::FunctionContext fn_ctx(cylon::ToArrowPool(ctx));
  arrow::compute::Datum local_result;
  arrow::Status status = arrow::compute::Sum(&fn_ctx, input, &local_result);

  if (status.ok()) {
    return DoAllReduce(ctx, local_result, output, data_type, cylon::ReduceOp::SUM);
  } else {
    LOG(ERROR) << "Local aggregation failed! " << status.message();
    return cylon::Status(Code::ExecutionError, status.message());
  }
}

cylon::Status Count(cylon::CylonContext *ctx,
                    const std::shared_ptr<cylon::Table> &table,
                    int32_t col_idx,
                    std::shared_ptr<Result> *output) {
  arrow::Status status;

  const shared_ptr<Column> &col = table->GetColumn(col_idx);
  const shared_ptr<DataType> &data_type = cylon::Int64();

  // count currently requires a single array (not a chunked array). So, merge arrays
  std::shared_ptr<arrow::Array> combined_input;
  if (!(arrow::Concatenate(col->GetColumnData()->chunks(), cylon::ToArrowPool(ctx), &combined_input)).ok()) {
    LOG(ERROR) << "Array concatenation failed! " << status.message();
    return cylon::Status(Code::ExecutionError, status.message());
  }
  const arrow::compute::Datum input(combined_input);

  arrow::compute::FunctionContext fn_ctx(cylon::ToArrowPool(ctx));
  arrow::compute::CountOptions options(arrow::compute::CountOptions::COUNT_ALL);
  arrow::compute::Datum local_result;
  status = arrow::compute::Count(&fn_ctx, options, input, &local_result);

  if (status.ok()) {
    return DoAllReduce(ctx, local_result, output, data_type, cylon::ReduceOp::SUM);
  } else {
    LOG(ERROR) << "Local aggregation failed! " << status.message();
    return cylon::Status(Code::ExecutionError, status.message());
  }
}

/**
 *
 * @tparam minMax set 0 for min and 1 for max
 * @param ctx
 * @param table
 * @param col_idx
 * @param output
 * @return
 */
template<bool minMax>
cylon::Status MinMax(cylon::CylonContext *ctx,
                     const std::shared_ptr<cylon::Table> &table,
                     int32_t col_idx,
                     std::shared_ptr<Result> *output) {
  const shared_ptr<Column> &col = table->GetColumn(col_idx);
  const shared_ptr<DataType> &data_type = col->GetDataType();
  const arrow::compute::Datum input(col->GetColumnData());

  arrow::compute::FunctionContext fn_ctx(cylon::ToArrowPool(ctx));
  arrow::compute::MinMaxOptions options;
  arrow::compute::Datum local_result; // minmax returns a vector<Datum>{min, max}
  arrow::Status status = arrow::compute::MinMax(&fn_ctx, options, input, &local_result);

  if (status.ok()) {
    return DoAllReduce(ctx, local_result.collection().at(minMax), output, data_type,
                       minMax ? cylon::ReduceOp::MIN : cylon::ReduceOp::MAX);
  } else {
    LOG(ERROR) << "Local aggregation failed! " << status.message();
    return cylon::Status(Code::ExecutionError, status.message());
  }
}

  cylon::Status Min(cylon::CylonContext *ctx,
                    const std::shared_ptr<cylon::Table> &table,
                    int32_t col_idx,
                    std::shared_ptr<Result> *output) {
    return MinMax<0>(ctx, table, col_idx, output);
  }

  cylon::Status Max(cylon::CylonContext *ctx,
                    const std::shared_ptr<cylon::Table> &table,
                    int32_t col_idx,
                    std::shared_ptr<Result> *output) {
    return MinMax<1>(ctx, table, col_idx, output);
  }

}  // end compute
} // end cylon


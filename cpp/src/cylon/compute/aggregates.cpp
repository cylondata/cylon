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

#include <status.hpp>
#include <table.hpp>
#include <ctx/arrow_memory_pool_utils.hpp>
#include <net/comm_operations.hpp>
#include <net/mpi/mpi_operations.hpp>

#include "compute/aggregates.hpp"
#include "util/macros.hpp"

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
        arrow::is_number_type<NUM_ARROW_T>::value | arrow::is_boolean_type<NUM_ARROW_T>::value>::type>
cylon::Status AllReduce(cylon::net::CommType comm_type,
                        const arrow::Datum &send,
                        std::shared_ptr<Result> &output,
                        const std::shared_ptr<DataType> &data_type,
                        cylon::net::ReduceOp reduce_op) {
  using NUM_ARROW_SCALAR_T = typename arrow::TypeTraits<NUM_ARROW_T>::ScalarType;

  const std::shared_ptr<NUM_ARROW_SCALAR_T> &send_scalar = std::static_pointer_cast<NUM_ARROW_SCALAR_T>(send.scalar());
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
      cylon::Status status;
      for (size_t i = 0; i < reduce_ops.size(); i++) {
        const std::shared_ptr<NUM_ARROW_SCALAR_T>
            &send_scalar =
            std::static_pointer_cast<NUM_ARROW_SCALAR_T>(send_struct_scalar->value[i]);
        std::shared_ptr<NUM_ARROW_SCALAR_T> rcv_scalar = std::make_shared<NUM_ARROW_SCALAR_T>();

        status = cylon::mpi::AllReduce(send_scalar->data(),
                                       rcv_scalar->mutable_data(),
                                       1,
                                       data_type,
                                       reduce_ops[i]);
        RETURN_IF_STATUS_FAILED(status)
        rcv_scalar_vector.push_back(rcv_scalar);
      }
      std::shared_ptr<arrow::StructScalar> rcv_struct_scalar
          = std::make_shared<arrow::StructScalar>(rcv_scalar_vector, send_struct_scalar->type);
      // build the output datum
      arrow::Datum global_result(rcv_struct_scalar);
      output = std::make_shared<Result>(global_result);

      return status;
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

//cylon::Status DoAllReduce(std::shared_ptr<cylon::CylonContext> &ctx,
//                          const arrow::Datum &snd,
//                          std::shared_ptr<Result> &rcv,
//                          const std::shared_ptr<DataType> &dtype,
//                          cylon::net::ReduceOp red_op) {
//  auto comm_type = ctx->GetCommType();
//  switch (dtype->getType()) {
//    case Type::BOOL:return AllReduce<arrow::BooleanType>(comm_type, snd, rcv, dtype, red_op);
//    case Type::UINT8:return AllReduce<arrow::UInt8Type>(comm_type, snd, rcv, dtype, red_op);
//    case Type::INT8:return AllReduce<arrow::Int8Type>(comm_type, snd, rcv, dtype, red_op);
//    case Type::UINT16:return AllReduce<arrow::UInt16Type>(comm_type, snd, rcv, dtype, red_op);
//    case Type::INT16:return AllReduce<arrow::Int16Type>(comm_type, snd, rcv, dtype, red_op);
//    case Type::UINT32:return AllReduce<arrow::UInt32Type>(comm_type, snd, rcv, dtype, red_op);
//    case Type::INT32:return AllReduce<arrow::Int32Type>(comm_type, snd, rcv, dtype, red_op);
//    case Type::UINT64:return AllReduce<arrow::UInt64Type>(comm_type, snd, rcv, dtype, red_op);
//    case Type::INT64:return AllReduce<arrow::Int64Type>(comm_type, snd, rcv, dtype, red_op);
//    case Type::FLOAT:return AllReduce<arrow::FloatType>(comm_type, snd, rcv, dtype, red_op);
//    case Type::DOUBLE:return AllReduce<arrow::DoubleType>(comm_type, snd, rcv, dtype, red_op);
//    case Type::HALF_FLOAT:break;
//    case Type::STRING:break;
//    case Type::BINARY:break;
//    case Type::FIXED_SIZE_BINARY:break;
//    case Type::DATE32:break;
//    case Type::DATE64:break;
//    case Type::TIMESTAMP:break;
//    case Type::TIME32:break;
//    case Type::TIME64:break;
//    case Type::INTERVAL:break;
//    case Type::DECIMAL:break;
//    case Type::LIST:break;
//    case Type::EXTENSION:break;
//    case Type::FIXED_SIZE_LIST:break;
//    case Type::DURATION:break;
//  }
//  return cylon::Status(cylon::Code::Invalid, "data type not supported!");
//}

cylon::Status Sum(const std::shared_ptr<cylon::Table> &table,
                  int32_t col_idx,
                  std::shared_ptr<Result> &output) {
  auto ctx = table->GetContext();
  const std::shared_ptr<Column> &col = table->GetColumn(col_idx); // cylon column object
  const std::shared_ptr<DataType> &data_type = col->GetDataType();
  const arrow::Datum input(col->GetColumnData()); // input datum

  // do local operation
  arrow::compute::ExecContext exec_ctx(cylon::ToArrowPool(ctx));
  arrow::Result<arrow::Datum> sum_res = arrow::compute::Sum(input, &exec_ctx);

  if (sum_res.ok()) {
    return DoAllReduce(ctx, sum_res.ValueOrDie(), output, data_type, cylon::net::ReduceOp::SUM);
  } else {
    const auto& status = sum_res.status();
    LOG(ERROR) << "Local aggregation failed! " << status.message();
    return cylon::Status(Code::ExecutionError, status.message());
  }
}

cylon::Status Count(const std::shared_ptr<cylon::Table> &table, int32_t col_idx, std::shared_ptr<Result> &output) {
  auto ctx = table->GetContext();

  const std::shared_ptr<Column> &col = table->GetColumn(col_idx);
  const std::shared_ptr<DataType> &data_type = cylon::Int64();
  const arrow::Datum input(col->GetColumnData()); // input datum

  arrow::compute::ExecContext exec_ctx(cylon::ToArrowPool(ctx));
  arrow::compute::CountOptions options(arrow::compute::CountOptions::COUNT_NON_NULL);
  const arrow::Result<arrow::Datum> &count_res = arrow::compute::Count(input, options, &exec_ctx);

  if (count_res.ok()) {
    return DoAllReduce(ctx, count_res.ValueOrDie(), output, data_type, cylon::net::ReduceOp::SUM);
  } else {
    const auto& status = count_res.status();
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
enum MinMaxOpts {
  min = 0,
  max,
  minmax
};

template<MinMaxOpts minMaxOpts>
cylon::Status min_max_impl(const std::shared_ptr<cylon::Table> &table,
                           int32_t col_idx,
                           std::shared_ptr<Result> &output) {
  auto ctx = table->GetContext();

  const std::shared_ptr<Column> &col = table->GetColumn(col_idx);
  const std::shared_ptr<DataType> &data_type = col->GetDataType();
  const arrow::Datum input(col->GetColumnData());

  arrow::compute::ExecContext exec_context(cylon::ToArrowPool(ctx));
  arrow::compute::MinMaxOptions options(arrow::compute::MinMaxOptions::SKIP);
  const arrow::Result<arrow::Datum> &result = arrow::compute::MinMax(input, options, &exec_context);

  if (result.ok()) {
    const arrow::Datum &local_result = result.ValueOrDie(); // minmax returns a structscalar
    const auto &struct_scalar = local_result.scalar_as<arrow::StructScalar>();

    if (minMaxOpts == min) {
      return DoAllReduce(ctx,
                         arrow::Datum(struct_scalar.value.at(0)),
                         output,
                         data_type,
                         cylon::net::ReduceOp::MIN);
    } else if (minMaxOpts == max) {
      return DoAllReduce(ctx,
                         arrow::Datum(struct_scalar.value.at(1)),
                         output,
                         data_type,
                         cylon::net::ReduceOp::MAX);
    } else {
      return DoAllReduce<std::vector<cylon::net::ReduceOp>>(ctx,
                                                            local_result,
                                                            output,
                                                            data_type,
                                                            {cylon::net::ReduceOp::MIN,
                                                             cylon::net::ReduceOp::MAX});
    }
  } else {
    const auto& status = result.status();
    LOG(ERROR) << "Local aggregation failed! " << status.message();
    return cylon::Status(Code::ExecutionError, status.message());
  }
}

cylon::Status Min(const std::shared_ptr<cylon::Table> &table,
                  int32_t col_idx,
                  std::shared_ptr<Result> &output) {
  return min_max_impl<MinMaxOpts::min>(table, col_idx, output);
}

cylon::Status Max(const std::shared_ptr<cylon::Table> &table,
                  int32_t col_idx,
                  std::shared_ptr<Result> &output) {
  return min_max_impl<MinMaxOpts::max>(table, col_idx, output);
}

cylon::Status MinMax(const std::shared_ptr<cylon::Table> &table,
                     int32_t col_idx,
                     std::shared_ptr<Result> &output) {
  return min_max_impl<MinMaxOpts::minmax>(table, col_idx, output);
}

template<typename ARROW_TYPE, typename = typename std::enable_if<
    arrow::is_number_type<ARROW_TYPE>::value
        | arrow::is_boolean_type<ARROW_TYPE>::value>::type>
cylon::Status ResolveTableFromScalar(const std::shared_ptr<cylon::Table> &input, int32_t col_idx,
                                     const std::shared_ptr<cylon::compute::Result> &result,
                                     std::shared_ptr<cylon::Table> &output) {
  using SCALAR_TYPE = typename arrow::TypeTraits<ARROW_TYPE>::ScalarType;
  using BUILDER_TYPE = typename arrow::TypeTraits<ARROW_TYPE>::BuilderType;

  std::shared_ptr<cylon::CylonContext> ctx = input->GetContext();

  arrow::Status s;
  std::vector<std::shared_ptr<arrow::Array>> out_vectors;
  BUILDER_TYPE idx_builder(cylon::ToArrowPool(ctx));
  std::shared_ptr<arrow::Array> out_idx;
  if (!(s = idx_builder.Reserve(1)).ok()) {
    return cylon::Status(Code::ExecutionError, s.message());
  }
  std::shared_ptr<SCALAR_TYPE>
      &&cast_sclr = std::static_pointer_cast<SCALAR_TYPE>(result->GetResult().scalar());

  idx_builder.UnsafeAppend(cast_sclr->value);

  if (!(s = idx_builder.Finish(&out_idx)).ok()) {
    return cylon::Status(Code::ExecutionError, s.message());
  }

  out_vectors.push_back(out_idx);

  std::shared_ptr<arrow::Table> arw_table;
  input->ToArrowTable(arw_table);

  auto out_a_table = arrow::Table::Make(arrow::schema({arw_table->schema()->field(col_idx)}), {out_vectors});

  return cylon::Table::FromArrowTable(ctx, out_a_table, output);
}

cylon::Status CreateTableFromScalar(const std::shared_ptr<cylon::Table> &input,
                                    int32_t col_idx,
                                    const std::shared_ptr<cylon::compute::Result> &result,
                                    std::shared_ptr<cylon::Table> &output) {

  switch (result->GetResult().scalar()->type->id()) {
    case arrow::Type::NA: break;
    case arrow::Type::BOOL: {
      return ResolveTableFromScalar<arrow::BooleanType>(input, col_idx, result, output);
    }
    case arrow::Type::UINT8: {
      return ResolveTableFromScalar<arrow::UInt8Type>(input, col_idx, result, output);
    }
    case arrow::Type::INT8: {
      return ResolveTableFromScalar<arrow::Int8Type>(input, col_idx, result, output);
    }
    case arrow::Type::UINT16: {
      return ResolveTableFromScalar<arrow::UInt16Type>(input, col_idx, result, output);
    }
    case arrow::Type::INT16: {
      return ResolveTableFromScalar<arrow::Int16Type>(input, col_idx, result, output);
    }
    case arrow::Type::UINT32: {
      return ResolveTableFromScalar<arrow::UInt32Type>(input, col_idx, result, output);
    }
    case arrow::Type::INT32: {
      return ResolveTableFromScalar<arrow::Int32Type>(input, col_idx, result, output);
    }
    case arrow::Type::UINT64: {
      return ResolveTableFromScalar<arrow::UInt64Type>(input, col_idx, result, output);
    }
    case arrow::Type::INT64: {
      return ResolveTableFromScalar<arrow::Int64Type>(input, col_idx, result, output);
    }
    case arrow::Type::FLOAT: {
      return ResolveTableFromScalar<arrow::FloatType>(input, col_idx, result, output);
    }
    case arrow::Type::DOUBLE: {
      return ResolveTableFromScalar<arrow::DoubleType>(input, col_idx, result, output);
    }
    case arrow::Type::HALF_FLOAT:break;
    case arrow::Type::STRING:break;
    case arrow::Type::BINARY:break;
    case arrow::Type::FIXED_SIZE_BINARY:break;
    case arrow::Type::DATE32:break;
    case arrow::Type::DATE64:break;
    case arrow::Type::TIMESTAMP:break;
    case arrow::Type::TIME32:break;
    case arrow::Type::TIME64:break;
    case arrow::Type::DECIMAL:break;
    case arrow::Type::LIST:break;
    case arrow::Type::STRUCT:break;
    case arrow::Type::DICTIONARY:break;
    case arrow::Type::MAP:break;
    case arrow::Type::EXTENSION:break;
    case arrow::Type::FIXED_SIZE_LIST:break;
    case arrow::Type::DURATION:break;
    case arrow::Type::LARGE_STRING:break;
    case arrow::Type::LARGE_BINARY:break;
    case arrow::Type::LARGE_LIST:break;
    case arrow::Type::INTERVAL_MONTHS:break;
    case arrow::Type::INTERVAL_DAY_TIME:break;
    case arrow::Type::SPARSE_UNION:break;
    case arrow::Type::DENSE_UNION:break;
    case arrow::Type::MAX_ID:break;
  }
  return cylon::Status(Code::NotImplemented, "Not Supported Type");
}

cylon::Status GenTable(cylon::compute::AggregateOperation aggregate_op,
                       const std::shared_ptr<cylon::Table> &input,
                       int32_t col_idx,
                       std::shared_ptr<cylon::Table> &output) {
  std::shared_ptr<cylon::compute::Result> result;

  cylon::Status s = aggregate_op(input, col_idx, result);

  if (s.is_ok()) {
    return CreateTableFromScalar(input, col_idx, result, output);
  } else {
    return s;
  }
}

cylon::Status Sum(const std::shared_ptr<cylon::Table> &table,
                  int32_t col_idx,
                  std::shared_ptr<cylon::Table> &output) {
  return GenTable(cylon::compute::Sum, table, col_idx, output);
}

cylon::Status Count(const std::shared_ptr<cylon::Table> &table,
                    int32_t col_idx,
                    std::shared_ptr<cylon::Table> &output) {
  return GenTable(cylon::compute::Count, table, col_idx, output);
}

cylon::Status Min(const std::shared_ptr<cylon::Table> &table,
                  int32_t col_idx,
                  std::shared_ptr<cylon::Table> &output) {
  return GenTable(cylon::compute::Min, table, col_idx, output);
}

cylon::Status Max(const std::shared_ptr<cylon::Table> &table,
                  int32_t col_idx,
                  std::shared_ptr<cylon::Table> &output) {
  return GenTable(cylon::compute::Max, table, col_idx, output);
}

}  // end compute
} // end cylon


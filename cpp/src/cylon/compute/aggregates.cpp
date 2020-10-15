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
#include <net/comm_operations.hpp>
#include <net/mpi/mpi_operations.hpp>

#include "compute/aggregates.hpp"

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
                        const arrow::compute::Datum &send,
                        std::shared_ptr<Result> *output,
                        const std::shared_ptr<DataType> &data_type,
                        cylon::net::ReduceOp reduce_op) {
  using NUM_ARROW_SCALAR_T = typename arrow::TypeTraits<NUM_ARROW_T>::ScalarType;

  const std::shared_ptr<NUM_ARROW_SCALAR_T> &send_scalar = std::static_pointer_cast<NUM_ARROW_SCALAR_T>(send.scalar());
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

cylon::Status DoAllReduce(std::shared_ptr<cylon::CylonContext> &ctx,
                          const arrow::compute::Datum &send,
                          std::shared_ptr<Result> *receive,
                          const std::shared_ptr<DataType> &data_type,
                          cylon::net::ReduceOp reduce_op) {
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

cylon::Status Sum(const std::shared_ptr<cylon::Table> &table, int32_t col_idx, std::shared_ptr<Result> *output) {
  auto ctx = table->GetContext();
  const std::shared_ptr<Column> &col = table->GetColumn(col_idx); // cylon column object
  const std::shared_ptr<DataType> &data_type = col->GetDataType();
  const arrow::compute::Datum input(col->GetColumnData()); // input datum

  // do local operation
  arrow::compute::FunctionContext fn_ctx(cylon::ToArrowPool(ctx));
  arrow::compute::Datum local_result;
  arrow::Status status = arrow::compute::Sum(&fn_ctx, input, &local_result);

  if (status.ok()) {
    return DoAllReduce(ctx, local_result, output, data_type, cylon::net::ReduceOp::SUM);
  } else {
    LOG(ERROR) << "Local aggregation failed! " << status.message();
    return cylon::Status(Code::ExecutionError, status.message());
  }
}

cylon::Status Count(const std::shared_ptr<cylon::Table> &table, int32_t col_idx, std::shared_ptr<Result> *output) {
  arrow::Status status;
  auto ctx = table->GetContext();

  const std::shared_ptr<Column> &col = table->GetColumn(col_idx);
  const std::shared_ptr<DataType> &data_type = cylon::Int64();

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
    return DoAllReduce(ctx, local_result, output, data_type, cylon::net::ReduceOp::SUM);
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
cylon::Status MinMax(const std::shared_ptr<cylon::Table> &table, int32_t col_idx, std::shared_ptr<Result> *output) {
  auto ctx = table->GetContext();

  const std::shared_ptr<Column> &col = table->GetColumn(col_idx);
  const std::shared_ptr<DataType> &data_type = col->GetDataType();
  const arrow::compute::Datum input(col->GetColumnData());

  arrow::compute::FunctionContext fn_ctx(cylon::ToArrowPool(ctx));
  arrow::compute::MinMaxOptions options;
  arrow::compute::Datum local_result; // minmax returns a vector<Datum>{min, max}
  arrow::Status status = arrow::compute::MinMax(&fn_ctx, options, input, &local_result);

  if (status.ok()) {
    return DoAllReduce(ctx, local_result.collection().at(minMax), output, data_type,
                       minMax ? cylon::net::ReduceOp::MIN : cylon::net::ReduceOp::MAX);
  } else {
    LOG(ERROR) << "Local aggregation failed! " << status.message();
    return cylon::Status(Code::ExecutionError, status.message());
  }
}

cylon::Status Min(const std::shared_ptr<cylon::Table> &table, int32_t col_idx, std::shared_ptr<Result> *output) {
  return MinMax<0>(table, col_idx, output);
}

cylon::Status Max(const std::shared_ptr<cylon::Table> &table, int32_t col_idx, std::shared_ptr<Result> *output) {
  return MinMax<1>(table, col_idx, output);
}

template<typename arrow_type, typename = typename std::enable_if<arrow::is_number_type<arrow_type>::value
                                                                     | arrow::is_boolean_type<arrow_type>::value>::type>
arrow::Status ResolveTableFromScalar(arrow::MemoryPool *pool,
                                     std::shared_ptr<cylon::CylonContext> &ctx,
                                     int32_t col_idx,
                                     std::shared_ptr<cylon::compute::Result> &result,
                                     std::shared_ptr<cylon::Table> input,
                                     std::shared_ptr<cylon::Table> &output) {
  using SCALAR_TYPE = typename arrow::TypeTraits<arrow_type>::ScalarType;
  using IDX_BUILDER_T = typename arrow::TypeTraits<arrow_type>::BuilderType;

  arrow::Status s;
  std::vector<std::shared_ptr<arrow::Array>> out_vectors;
  IDX_BUILDER_T idx_builder(pool);
  std::shared_ptr<arrow::Array> out_idx;
  if (!(s = idx_builder.Reserve(1)).ok()) {
    return s;
  }
  std::shared_ptr<SCALAR_TYPE>
      &&cast_sclr = std::static_pointer_cast<SCALAR_TYPE>(result->GetResult().scalar());

  idx_builder.UnsafeAppend(cast_sclr->value);

  if (!(s = idx_builder.Finish(&out_idx)).ok()) {
    return s;
  }

  out_vectors.push_back(out_idx);

  std::shared_ptr<arrow::Table> arw_table;
  input->ToArrowTable(arw_table);

  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::shared_ptr<arrow::Field> selected_filed = arw_table->schema()->field(col_idx);

  fields.push_back(std::make_shared<arrow::Field>(selected_filed->name(),
                                                  selected_filed->type(),
                                                  selected_filed->nullable()));

  auto schema = arrow::schema(fields);
  auto out_a_table = arrow::Table::Make(schema, {out_vectors});

  Status status = cylon::Table::FromArrowTable(ctx, out_a_table, &output);

  if (status.is_ok()) {
    return arrow::Status().OK();
  } else {
    return arrow::Status::ExecutionError("Table resolution failed in aggregation");
  }
}

typedef arrow::Status
(*ResolveTableFromScalarFptr)(arrow::MemoryPool *pool,
                              std::shared_ptr<cylon::CylonContext> &ctx,
                              int32_t col_idx,
                              std::shared_ptr<cylon::compute::Result> &result,
                              std::shared_ptr<cylon::Table> input,
                              std::shared_ptr<cylon::Table> &output);
ResolveTableFromScalarFptr CreateTableFromScalar(std::shared_ptr<cylon::compute::Result> &result) {

  switch (result->GetResult().scalar()->type->id()) {

    case arrow::Type::NA: {
      return nullptr;
    }
    case arrow::Type::BOOL: {
      return &ResolveTableFromScalar<arrow::BooleanType>;
    }
    case arrow::Type::UINT8: {
      return &ResolveTableFromScalar<arrow::UInt8Type>;
    }
    case arrow::Type::INT8: {
      return &ResolveTableFromScalar<arrow::Int8Type>;
    }
    case arrow::Type::UINT16: {
      return &ResolveTableFromScalar<arrow::UInt16Type>;
    }
    case arrow::Type::INT16: {
      return &ResolveTableFromScalar<arrow::Int16Type>;
    }
    case arrow::Type::UINT32: {
      return &ResolveTableFromScalar<arrow::UInt32Type>;
    }
    case arrow::Type::INT32: {
      return &ResolveTableFromScalar<arrow::Int32Type>;
    }
    case arrow::Type::UINT64: {
      return &ResolveTableFromScalar<arrow::UInt64Type>;
    }
    case arrow::Type::INT64: {
      return &ResolveTableFromScalar<arrow::Int64Type>;
    }
    case arrow::Type::FLOAT: {
      return &ResolveTableFromScalar<arrow::FloatType>;
    }
    case arrow::Type::DOUBLE: {
      return &ResolveTableFromScalar<arrow::DoubleType>;
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
    case arrow::Type::INTERVAL:break;
    case arrow::Type::DECIMAL:break;
    case arrow::Type::LIST:break;
    case arrow::Type::STRUCT:break;
    case arrow::Type::UNION:break;
    case arrow::Type::DICTIONARY:break;
    case arrow::Type::MAP:break;
    case arrow::Type::EXTENSION:break;
    case arrow::Type::FIXED_SIZE_LIST:break;
    case arrow::Type::DURATION:break;
    case arrow::Type::LARGE_STRING:break;
    case arrow::Type::LARGE_BINARY:break;
    case arrow::Type::LARGE_LIST:break;
  }

  return nullptr;
}

cylon::Status Sum(const std::shared_ptr<cylon::Table> &table,
                  int32_t col_idx,
                  std::shared_ptr<cylon::Table> &output) {

  cylon::Status status;
  std::shared_ptr<cylon::compute::Result> result;
  std::shared_ptr<cylon::CylonContext> ctx = table->GetContext();

  if ((status = cylon::compute::Sum(table, col_idx, &result)).is_ok()) {
    ResolveTableFromScalarFptr fptr = CreateTableFromScalar(result);
    arrow::MemoryPool *memory_pool = cylon::ToArrowPool(ctx);
    fptr(memory_pool, ctx, col_idx, result, table, output);
  }
  return status;
}

cylon::Status Count(const std::shared_ptr<cylon::Table> &table,
                             int32_t col_idx,
                             std::shared_ptr<cylon::Table> &output) {
  cylon::Status status;
  std::shared_ptr<cylon::compute::Result> result;
  std::shared_ptr<cylon::CylonContext> ctx = table->GetContext();

  if ((status = cylon::compute::Count(table, col_idx, &result)).is_ok()) {
    ResolveTableFromScalarFptr fptr = CreateTableFromScalar(result);
    arrow::MemoryPool *memory_pool = cylon::ToArrowPool(ctx);
    fptr(memory_pool, ctx, col_idx, result, table, output);
  }
  return status;
}

cylon::Status Min(const std::shared_ptr<cylon::Table> &table,
                           int32_t col_idx,
                           std::shared_ptr<cylon::Table> &output) {
  cylon::Status status;
  std::shared_ptr<cylon::compute::Result> result;
  std::shared_ptr<cylon::CylonContext> ctx = table->GetContext();

  if ((status = cylon::compute::Min(table, col_idx, &result)).is_ok()) {
    ResolveTableFromScalarFptr fptr = CreateTableFromScalar(result);
    arrow::MemoryPool *memory_pool = cylon::ToArrowPool(ctx);
    fptr(memory_pool, ctx, col_idx, result, table, output);
  }
  return status;
}

cylon::Status Max(const std::shared_ptr<cylon::Table> &table,
                           int32_t col_idx,
                           std::shared_ptr<cylon::Table> &output) {
  cylon::Status status;
  std::shared_ptr<cylon::compute::Result> result;
  std::shared_ptr<cylon::CylonContext> ctx = table->GetContext();

  if ((status = cylon::compute::Max(table, col_idx, &result)).is_ok()) {
    ResolveTableFromScalarFptr fptr = CreateTableFromScalar(result);
    arrow::MemoryPool *memory_pool = cylon::ToArrowPool(ctx);
    fptr(memory_pool, ctx, col_idx, result, table, output);
  }
  return status;
}

}  // end compute
} // end cylon


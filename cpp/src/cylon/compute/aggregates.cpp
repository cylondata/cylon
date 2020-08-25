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
#include "compute/aggregates.hpp"

namespace cylon {
namespace compute {

/**
 * Container to resolve the arrow and cylon type templates
 */
struct DistAggOp {
  virtual int Execute(const shared_ptr<SyncChannel> &reduce_channel,
                      const arrow::compute::Datum &send,
                      arrow::compute::Datum *receive,
                      const shared_ptr<DataType> &data_type,
                      cylon::ReduceOp reduce_op) = 0;
};

/**
 * DistAggOp implementation for arrow numeric types
 * @tparam NUM_ARROW_T numeric arrow type ()
 */
template<typename NUM_ARROW_T,
    typename = typename std::enable_if<
        arrow::is_number_type<NUM_ARROW_T>::value | arrow::is_boolean_type<NUM_ARROW_T>::value>::type>
struct NumericDistAggOp : DistAggOp {
  using NUM_ARROW_SCALAR_T = typename arrow::TypeTraits<NUM_ARROW_T>::ScalarType;

  int Execute(const shared_ptr<SyncChannel> &reduce_channel,
              const arrow::compute::Datum &send,
              arrow::compute::Datum *receive,
              const shared_ptr<DataType> &data_type,
              cylon::ReduceOp reduce_op) override {

    const shared_ptr<NUM_ARROW_SCALAR_T> &send_scalar = std::static_pointer_cast<NUM_ARROW_SCALAR_T>(send.scalar());
    std::shared_ptr<NUM_ARROW_SCALAR_T> recv_scalar = std::make_shared<NUM_ARROW_SCALAR_T>();

    if (reduce_channel->AllReduce(&(send_scalar->value), &(recv_scalar->value), send.length(), data_type, reduce_op)
        == 0) {
      *receive = arrow::compute::Datum(recv_scalar);
      return 0;
    } else {
      return 1;
    }
  };
};

std::shared_ptr<DistAggOp> GetDistAggOp(const shared_ptr<DataType> &data_type) {
  switch (data_type->getType()) {
    case Type::BOOL: return std::make_shared<NumericDistAggOp<arrow::BooleanType>>();
    case Type::UINT8:return std::make_shared<NumericDistAggOp<arrow::UInt8Type>>();
    case Type::INT8:return std::make_shared<NumericDistAggOp<arrow::Int8Type>>();
    case Type::UINT16:return std::make_shared<NumericDistAggOp<arrow::UInt16Type>>();
    case Type::INT16:return std::make_shared<NumericDistAggOp<arrow::Int16Type>>();
    case Type::UINT32:return std::make_shared<NumericDistAggOp<arrow::UInt32Type>>();
    case Type::INT32:return std::make_shared<NumericDistAggOp<arrow::Int32Type>>();
    case Type::UINT64:return std::make_shared<NumericDistAggOp<arrow::UInt64Type>>();
    case Type::INT64:return std::make_shared<NumericDistAggOp<arrow::Int64Type>>();
    case Type::FLOAT:return std::make_shared<NumericDistAggOp<arrow::FloatType>>();
    case Type::DOUBLE:return std::make_shared<NumericDistAggOp<arrow::DoubleType>>();
    case Type::HALF_FLOAT: break;
    case Type::STRING:break;
    case Type::BINARY:break;
    case Type::FIXED_SIZE_BINARY:break;
    case Type::DATE32:break;
    case Type::DATE64:break;
    case Type::TIMESTAMP:break;
    case Type::TIME32:break;
    case Type::TIME64:break;
    case Type::INTERVAL:break;
    case Type::DECIMAL:break;
    case Type::LIST:break;
    case Type::EXTENSION:break;
    case Type::FIXED_SIZE_LIST:break;
    case Type::DURATION:break;
  }
  return nullptr;
}

/**
 * Method to run the distributed aggregations after the local operations
 * @param ctx
 * @param data_type
 * @param local_result
 * @param output
 * @param global_reduce_op
 * @return
 */
cylon::Status DoDistributedAggregation(cylon::CylonContext *ctx,
                                       const shared_ptr<DataType> &data_type,
                                       const arrow::compute::Datum &local_result,
                                       std::shared_ptr<Result> *output,
                                       cylon::ReduceOp global_reduce_op) {
  if (ctx->IsDistributed()) {
    arrow::compute::Datum global_result;

    const shared_ptr<SyncChannel> &reduce_channel = ctx->GetCommunicator()->MakeReduceChannel();
    const shared_ptr<DistAggOp> &dist_agg_op = GetDistAggOp(data_type);

    if (dist_agg_op->Execute(reduce_channel, local_result, &global_result, data_type, global_reduce_op)) {
      return cylon::Status(Code::ExecutionError, "Distribution aggregation failed!");

    } else {
      *output = std::make_shared<Result>(global_result);
    }
  } else { // return local result
    *output = std::make_shared<Result>(local_result);
  }

  return cylon::Status::OK();
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
    return DoDistributedAggregation(ctx, data_type, local_result, output, cylon::ReduceOp::SUM);
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
    return DoDistributedAggregation(ctx, data_type, local_result, output, cylon::ReduceOp::SUM);
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
    return DoDistributedAggregation(ctx, data_type, local_result.collection().at(minMax), output,
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


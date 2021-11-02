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

#include <cylon/net/comm_operations.hpp>
#include <cylon/util/macros.hpp>

#include <cylon/compute/aggregates.hpp>
#include <cylon/compute/aggregate_utils.hpp>

namespace cylon {
namespace compute {

cylon::Status Sum(const std::shared_ptr<cylon::Table> &table,
                  int32_t col_idx,
                  std::shared_ptr<Result> &output) {
  const auto &ctx = table->GetContext();
  const auto &a_table = table->get_table();
  const auto &a_col = a_table->column(col_idx);
  const auto &dtype = tarrow::ToCylonType(a_col->type());

  // do local operation
  arrow::compute::ExecContext exec_ctx(cylon::ToArrowPool(ctx));
  CYLON_ASSIGN_OR_RAISE(auto sum_res, arrow::compute::Sum(a_col, &exec_ctx));

  // arrow sum upcasts sum to Int64, UInt64, Float64. So, change back to original type
  CYLON_ASSIGN_OR_RAISE(auto cast_res, arrow::compute::Cast(sum_res, a_col->type()));

  if (ctx->GetWorldSize() > 1) {
    return DoAllReduce(ctx, cast_res, output, dtype, cylon::net::ReduceOp::SUM);
  } else {
    output = std::make_shared<Result>(std::move(cast_res));
    return Status::OK();
  }
}

cylon::Status Count(const std::shared_ptr<cylon::Table> &table, int32_t col_idx, std::shared_ptr<Result> &output) {
  const auto &ctx = table->GetContext();
  const auto &a_table = table->get_table();
  const auto &a_col = a_table->column(col_idx);
  const auto &data_type = cylon::Int64();

  arrow::compute::ExecContext exec_ctx(cylon::ToArrowPool(ctx));
  arrow::compute::CountOptions options(arrow::compute::CountOptions::COUNT_NON_NULL);
  CYLON_ASSIGN_OR_RAISE(auto count_res, arrow::compute::Count(a_col, options, &exec_ctx));

  if (ctx->GetWorldSize() > 1) {
    return DoAllReduce(ctx, count_res, output, data_type, cylon::net::ReduceOp::SUM);
  } else {
    output = std::make_shared<Result>(count_res);
    return Status::OK();
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
cylon::Status static inline min_max_impl(const std::shared_ptr<CylonContext> &ctx,
                                         const arrow::Datum &input,
                                         const std::shared_ptr<cylon::DataType> &data_type,
                                         std::shared_ptr<Result> &output) {
  if (!input.is_arraylike()) {
    LOG_AND_RETURN_ERROR(Code::Invalid, "input should be array like");
  }

  arrow::compute::ExecContext exec_context(cylon::ToArrowPool(ctx));
  arrow::compute::MinMaxOptions options(arrow::compute::MinMaxOptions::SKIP);
  CYLON_ASSIGN_OR_RAISE(auto result, arrow::compute::MinMax(input, options, &exec_context));

  const auto &struct_scalar = result.scalar_as<arrow::StructScalar>();

  switch (minMaxOpts) {
    case min:
      return DoAllReduce(ctx,
                         arrow::Datum(struct_scalar.value.at(0)),
                         output,
                         data_type,
                         cylon::net::ReduceOp::MIN);
    case max:
      return DoAllReduce(ctx,
                         arrow::Datum(struct_scalar.value.at(1)),
                         output,
                         data_type,
                         cylon::net::ReduceOp::MAX);
    case minmax:
      return DoAllReduce<std::vector<cylon::net::ReduceOp>>(ctx,
                                                            result,
                                                            output,
                                                            data_type,
                                                            {cylon::net::ReduceOp::MIN,
                                                             cylon::net::ReduceOp::MAX});
  }
  return Status::OK(); // this would never reach
}

cylon::Status Min(const std::shared_ptr<cylon::Table> &table,
                  int32_t col_idx,
                  std::shared_ptr<Result> &output) {
  const auto &ctx = table->GetContext();
  const auto &a_table = table->get_table();
  const auto &a_col = a_table->column(col_idx);
  const auto &dtype = tarrow::ToCylonType(a_col->type());

  return min_max_impl<MinMaxOpts::min>(ctx, a_col, dtype, output);
}

cylon::Status Max(const std::shared_ptr<cylon::Table> &table,
                  int32_t col_idx,
                  std::shared_ptr<Result> &output) {
  const auto &ctx = table->GetContext();
  const auto &a_table = table->get_table();
  const auto &a_col = a_table->column(col_idx);
  const auto &dtype = tarrow::ToCylonType(a_col->type());
  return min_max_impl<MinMaxOpts::max>(ctx, a_col, dtype, output);
}

cylon::Status MinMax(const std::shared_ptr<cylon::Table> &table,
                     int32_t col_idx,
                     std::shared_ptr<Result> &output) {
  const auto &ctx = table->GetContext();
  const auto &a_table = table->get_table();
  const auto &a_col = a_table->column(col_idx);
  const auto &dtype = tarrow::ToCylonType(a_col->type());
  return min_max_impl<MinMaxOpts::minmax>(ctx, a_col, dtype, output);
}

cylon::Status MinMax(const std::shared_ptr<CylonContext> &ctx,
                     const arrow::Datum &array,
                     const std::shared_ptr<cylon::DataType> &datatype,
                     std::shared_ptr<Result> &output) {
  return min_max_impl<MinMaxOpts::minmax>(ctx, array, datatype, output);
}

template<typename ARROW_TYPE, typename = typename std::enable_if<
    arrow::is_number_type<ARROW_TYPE>::value
        | arrow::is_boolean_type<ARROW_TYPE>::value | arrow::is_temporal_type<ARROW_TYPE>::value>::type>
cylon::Status ResolveTableFromScalar(const std::shared_ptr<cylon::Table> &input, int32_t col_idx,
                                     const std::shared_ptr<cylon::compute::Result> &result,
                                     std::shared_ptr<cylon::Table> &output) {
  using SCALAR_TYPE = typename arrow::TypeTraits<ARROW_TYPE>::ScalarType;
  using BUILDER_TYPE = typename arrow::TypeTraits<ARROW_TYPE>::BuilderType;

  const auto& ctx = input->GetContext();

  arrow::Status s;
  std::vector<std::shared_ptr<arrow::Array>> out_vectors;

  std::shared_ptr<arrow::Table> arw_table;
  input->ToArrowTable(arw_table);

  BUILDER_TYPE idx_builder(arw_table->column(col_idx)->type(), cylon::ToArrowPool(ctx));

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
    case arrow::Type::DATE32:{
      return ResolveTableFromScalar<arrow::Date32Type>(input, col_idx, result, output);
    }
    case arrow::Type::DATE64:{
      return ResolveTableFromScalar<arrow::Date64Type>(input, col_idx, result, output);
    }
    case arrow::Type::TIMESTAMP:{
      return ResolveTableFromScalar<arrow::TimestampType>(input, col_idx, result, output);
    }
    case arrow::Type::TIME32:{
      return ResolveTableFromScalar<arrow::Time32Type>(input, col_idx, result, output);
    }
    case arrow::Type::TIME64:{
      return ResolveTableFromScalar<arrow::Time64Type>(input, col_idx, result, output);
    }
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
    case arrow::Type::DECIMAL256:break;
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


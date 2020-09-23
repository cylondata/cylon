//
// Created by niranda on 9/17/20.
//

#ifndef CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_HPP_
#define CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_HPP_

#include <status.hpp>
#include <table.hpp>
#include <compute/aggregates.hpp>
#include <map>

#include "groupby_aggregate_ops.hpp"
#include "groupby_hash.hpp"

namespace cylon {

#include "groupby_aggregate_ops.hpp"
#include "groupby_hash.hpp"

enum GroupByAlgorithm{
  HASH,
  PIPELINE
};

//Status GroupBy(const std::shared_ptr<Table> &table,
//               int64_t index_col,
//               const std::vector<int64_t> &aggregate_cols,
//               const std::vector<GroupByAggregationOp> &aggregate_ops,
//               std::shared_ptr<Table> &output);


/**
  * Local group by operation
  * Restrictions:
  *  - 0th col is the index col
  *  - every column has an aggregation op
  * @tparam IDX_ARROW_T index column type
  * @param table
  * @param index_col
  * @param aggregate_cols
  * @param aggregate_ops
  * @param output
  * @return
  */
template<typename IDX_ARROW_T,
    typename = typename std::enable_if<
        arrow::is_number_type<IDX_ARROW_T>::value | arrow::is_boolean_type<IDX_ARROW_T>::value>::type>
cylon::Status LocalGroupBy(const std::shared_ptr<cylon::Table> &table,
                           const std::vector<cylon::GroupByAggregationOp> &aggregate_ops,
                           std::shared_ptr<cylon::Table> &output) {
  if ((std::size_t) table->Columns() != aggregate_ops.size() + 1)
    return cylon::Status(cylon::Code::Invalid, "num cols != aggergate ops + 1");

  auto ctx = table->GetContext();
  auto a_table = table->get_table();

  arrow::Status a_status;
  arrow::MemoryPool *memory_pool = cylon::ToArrowPool(ctx);

  const int cols = a_table->num_columns();
  const std::shared_ptr<arrow::ChunkedArray> &idx_col = a_table->column(0);

  std::vector<shared_ptr<arrow::Array>> out_vectors;
  cylon::Status status;
  for (int c = 1; c < cols; c++) {
    const shared_ptr<arrow::ChunkedArray> &val_col = a_table->column(c);
    const shared_ptr<DataType> &val_data_type = table->GetColumn(c)->GetDataType();

    const HashGroupByFptr hash_group_by = PickHashGroupByFptr<IDX_ARROW_T>(val_data_type, aggregate_ops[c - 1]);

    status = hash_group_by(memory_pool, idx_col, val_col, out_vectors);
    if (!status.is_ok()){
      LOG(FATAL) << "Aggregation failed!";
      return status;
    }
  }

  auto out_a_table = arrow::Table::Make(a_table->schema(), out_vectors);

  return cylon::Table::FromArrowTable(ctx, out_a_table, &output);
}

/**
 * Pick a local group by function based on the index column data type
 * @param idx_data_type
 * @return
 */
typedef Status
(*LocalGroupByFptr)(const std::shared_ptr<Table> &table,
                    const std::vector<cylon::GroupByAggregationOp> &aggregate_ops,
                    std::shared_ptr<Table> &output);
LocalGroupByFptr PickLocalGroupByFptr(const shared_ptr<cylon::DataType> &idx_data_type) {
  switch (idx_data_type->getType()) {
    case Type::BOOL: return &LocalGroupBy<arrow::BooleanType>;
    case Type::UINT8: return &LocalGroupBy<arrow::UInt8Type>;
    case Type::INT8: return &LocalGroupBy<arrow::Int8Type>;
    case Type::UINT16: return &LocalGroupBy<arrow::UInt16Type>;
    case Type::INT16: return &LocalGroupBy<arrow::Int16Type>;
    case Type::UINT32: return &LocalGroupBy<arrow::UInt32Type>;
    case Type::INT32: return &LocalGroupBy<arrow::Int32Type>;
    case Type::UINT64: return &LocalGroupBy<arrow::UInt64Type>;
    case Type::INT64: return &LocalGroupBy<arrow::Int64Type>;
    case Type::FLOAT: return &LocalGroupBy<arrow::FloatType>;
    case Type::DOUBLE: return &LocalGroupBy<arrow::DoubleType>;
    case Type::HALF_FLOAT:break;
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

cylon::Status GroupBy(const std::shared_ptr<Table> &table,
                      int64_t index_col,
                      const std::vector<int64_t> &aggregate_cols,
                      const std::vector<cylon::GroupByAggregationOp> &aggregate_ops,
                      std::shared_ptr<Table> &output) {
  Status status;
  LocalGroupByFptr group_by_fptr = PickLocalGroupByFptr(table->GetColumn(index_col)->GetDataType());

  // first filter aggregation cols
  std::vector<int64_t> project_cols = {index_col};
  project_cols.insert(project_cols.end(), aggregate_cols.begin(), aggregate_cols.end());

  shared_ptr<Table> projected_table;
  if (!(status = table->Project(project_cols, projected_table)).is_ok()) {
    LOG(FATAL) << "table projection failed! " << status.get_msg();
    return status;
  }

  return group_by_fptr(projected_table, aggregate_ops, output);
}

}

#endif //CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_HPP_

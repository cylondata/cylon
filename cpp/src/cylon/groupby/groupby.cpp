//
// Created by niranda on 9/17/20.
//

#include "groupby.hpp"
#include <join/join_utils.hpp>
#include <util/arrow_utils.hpp>

namespace cylon {

//template<typename ARROW_T, typename C_TYPE>
//cylon::Status ProcessGroup(const std::shared_ptr<Table> &table, const C_TYPE &start, const C_TYPE &end) {
//  return Status::OK();
//}

//template<typename ARROW_ARRAY_T, typename C_TYPE>
//void GenerateGroupedColumns(const shared_ptr<ARROW_ARRAY_T> col, std::vector<int64_t> &boundaries) {
//  const int64_t num_rows = col->length();
//
//  if (!num_rows) return;
//
//  boundaries.clear();
//  boundaries.push_back(0);
//
//  C_TYPE prev_val = col->Value(0), curr_val;
//  for (int64_t i = 1; i < num_rows; ++i) {
//    curr_val = col->Value(i);
//
//    if (curr_val != prev_val) {
////      ProcessGroup<>(start_idx, i);
//      prev_val = curr_val;
////      prev_idx = i + 1; // max would be num_rows. so, no OOB
//    }
//  }
//  boundaries.push_back(num_rows);
//}

/**
 * Local group by operation
 * Restrictions:
 *  - 0th col is the index col 
 *  - every column has an aggregation op
 * @tparam NUM_ARROW_T
 * @param table
 * @param index_col
 * @param aggregate_cols
 * @param aggregate_ops
 * @param output
 * @return
 */
template<typename NUM_ARROW_T,
    typename = typename std::enable_if<
        arrow::is_number_type<NUM_ARROW_T>::value | arrow::is_boolean_type<NUM_ARROW_T>::value>::type>
cylon::Status LocalGroupBy(const std::shared_ptr<Table> &table,
                           const std::vector<cylon::compute::AggregateOperation> &aggregate_ops,
                           std::shared_ptr<Table> &output) {
  if ((std::size_t) table->Columns() != aggregate_ops.size() + 1)
    return cylon::Status(cylon::Code::Invalid, "num cols != aggergate ops + 1");

  auto ctx = table->GetContext();

  shared_ptr<arrow::Table> sorted_table;
  arrow::Status a_status;
  arrow::MemoryPool *memory_pool = cylon::ToArrowPool(ctx);

  // sort table (+ combining table chunks)
  a_status = cylon::util::SortTable(table->get_table(), 0, &sorted_table, memory_pool);
  if (!a_status.ok()) {
    LOG(FATAL) << "Failed to sort column to indices" << a_status.ToString();
    return cylon::Status(cylon::Code::ExecutionError, a_status.message());
  }

//  using ARROW_ARRAY_T = typename arrow::TypeTraits<NUM_ARROW_T>::ArrayType; // arrow numeric type of the
//  using C_TYPE = typename arrow::TypeTraits<NUM_ARROW_T>::CType;
//
//  // builders for new columns
//  std::vector<std::unique_ptr<arrow::ArrayBuilder>> builders;
//  for (auto &&col: sorted_table->columns()) {
//    std::unique_ptr<arrow::ArrayBuilder> builder;
//    if (!(a_status = arrow::MakeBuilder(memory_pool, col->type(), &builder)).ok()) {
//      return cylon::Status(cylon::Code::ExecutionError, a_status.message());
//    }
//    builders.push_back(builder);
//  }
//
//  const shared_ptr<ARROW_ARRAY_T>
//      &index_col = std::static_pointer_cast<ARROW_ARRAY_T>(sorted_table->column(0)->chunk(0));

  return cylon::Status();
}

/**
 * function pointer for local groupby operators 
 */
typedef Status
(*LocalGroupByFptr)(const std::shared_ptr<Table> &table,
                    const std::vector<cylon::compute::AggregateOperation> &aggregate_ops,
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
                      const std::vector<cylon::compute::AggregateOperation> &aggregate_ops,
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
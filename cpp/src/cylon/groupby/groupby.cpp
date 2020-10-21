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

#include <util/arrow_utils.hpp>

#include "groupby_hash.hpp"
#include "groupby_pipeline.hpp"
#include "groupby.hpp"

namespace cylon {

/**
 * Pick a local group by function based on the index column data type
 * @param idx_data_type 
 * @return 
 */
typedef Status
(*LocalGroupByFptr)(const std::shared_ptr<Table> &table,
                    const std::vector<cylon::GroupByAggregationOp> &aggregate_ops,
                    std::shared_ptr<Table> &output);
LocalGroupByFptr PickLocalHashGroupByFptr(const std::shared_ptr<cylon::DataType> &idx_data_type) {
  switch (idx_data_type->getType()) {
    case Type::BOOL: return &LocalHashGroupBy<arrow::BooleanType>;
    case Type::UINT8: return &LocalHashGroupBy<arrow::UInt8Type>;
    case Type::INT8: return &LocalHashGroupBy<arrow::Int8Type>;
    case Type::UINT16: return &LocalHashGroupBy<arrow::UInt16Type>;
    case Type::INT16: return &LocalHashGroupBy<arrow::Int16Type>;
    case Type::UINT32: return &LocalHashGroupBy<arrow::UInt32Type>;
    case Type::INT32: return &LocalHashGroupBy<arrow::Int32Type>;
    case Type::UINT64: return &LocalHashGroupBy<arrow::UInt64Type>;
    case Type::INT64: return &LocalHashGroupBy<arrow::Int64Type>;
    case Type::FLOAT: return &LocalHashGroupBy<arrow::FloatType>;
    case Type::DOUBLE: return &LocalHashGroupBy<arrow::DoubleType>;
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

LocalGroupByFptr PickLocalPipelineGroupByFptr(const std::shared_ptr<cylon::DataType> &idx_data_type) {
  switch (idx_data_type->getType()) {
    case Type::BOOL: return &LocalPipelinedGroupBy<arrow::BooleanType>;
    case Type::UINT8: return &LocalPipelinedGroupBy<arrow::UInt8Type>;
    case Type::INT8: return &LocalPipelinedGroupBy<arrow::Int8Type>;
    case Type::UINT16: return &LocalPipelinedGroupBy<arrow::UInt16Type>;
    case Type::INT16: return &LocalPipelinedGroupBy<arrow::Int16Type>;
    case Type::UINT32: return &LocalPipelinedGroupBy<arrow::UInt32Type>;
    case Type::INT32: return &LocalPipelinedGroupBy<arrow::Int32Type>;
    case Type::UINT64: return &LocalPipelinedGroupBy<arrow::UInt64Type>;
    case Type::INT64: return &LocalPipelinedGroupBy<arrow::Int64Type>;
    case Type::FLOAT: return &LocalPipelinedGroupBy<arrow::FloatType>;
    case Type::DOUBLE: return &LocalPipelinedGroupBy<arrow::DoubleType>;
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

cylon::Status GroupBy(std::shared_ptr<Table> &table,
                      int64_t index_col,
                      const std::vector<int64_t> &aggregate_cols,
                      const std::vector<cylon::GroupByAggregationOp> &aggregate_ops,
                      std::shared_ptr<Table> &output) {
  LocalGroupByFptr group_by_fptr = PickLocalHashGroupByFptr(table->GetColumn(index_col)->GetDataType());

  Status status;

  // first filter aggregation cols
  std::vector<int64_t> project_cols = {index_col};
  project_cols.insert(project_cols.end(), aggregate_cols.begin(), aggregate_cols.end());

  std::shared_ptr<Table> projected_table;
  auto t1 = std::chrono::high_resolution_clock::now();
  if (!(status = cylon::Project(table, project_cols, projected_table)).is_ok()) {
    LOG(FATAL) << "table projection failed! " << status.get_msg();
    return status;
  }
  auto t2 = std::chrono::high_resolution_clock::now();

  // do local group by
  std::shared_ptr<Table> local_table;
  if (!(status = group_by_fptr(projected_table, aggregate_ops, local_table)).is_ok()) {
    LOG(FATAL) << "Local group by failed! " << status.get_msg();
    return status;
  }
  auto t3 = std::chrono::high_resolution_clock::now();

  if (table->GetContext()->GetWorldSize() > 1) {
    // shuffle
    if (!(status = cylon::Shuffle(local_table, {0}, local_table)).is_ok()) {
      LOG(FATAL) << " table shuffle failed! " << status.get_msg();
      return status;
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    // do local distribute again
    if (!(status = group_by_fptr(local_table, aggregate_ops, output)).is_ok()) {
      LOG(FATAL) << "Local group by failed! " << status.get_msg();
      return status;
    }
    auto t5 = std::chrono::high_resolution_clock::now();

    LOG(INFO) << "groupby times "
              << " p " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " l " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
              << " s " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()
              << " l " << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count()
              << " t " << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t1).count();
  } else {
    output = local_table;
    LOG(INFO) << "groupby times "
              << " p " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " l " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
              << " s 0 l 0"
              << " t " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t1).count();
  }

  return Status::OK();
}

Status PipelineGroupBy(std::shared_ptr<Table> &table,
                       int64_t index_col,
                       const std::vector<int64_t> &aggregate_cols,
                       const std::vector<GroupByAggregationOp> &aggregate_ops,
                       std::shared_ptr<Table> &output) {
  LocalGroupByFptr group_by_fptr = PickLocalPipelineGroupByFptr(table->GetColumn(index_col)->GetDataType());

  Status status;

  // first filter aggregation cols
  std::vector<int64_t> project_cols = {index_col};
  project_cols.insert(project_cols.end(), aggregate_cols.begin(), aggregate_cols.end());

  std::shared_ptr<Table> projected_table;
  if (!(status = cylon::Project(table, project_cols, projected_table)).is_ok()) {
    LOG(FATAL) << "table projection failed! " << status.get_msg();
    return status;
  }

  // do local group by
  std::shared_ptr<Table> local_table;
  if (!(status = group_by_fptr(projected_table, aggregate_ops, local_table)).is_ok()) {
    LOG(FATAL) << "Local group by failed! " << status.get_msg();
    return status;
  }

  if (table->GetContext()->GetWorldSize() > 1) {
    // shuffle
    if (!(status = cylon::Shuffle(local_table, {0}, local_table)).is_ok()) {
      LOG(FATAL) << " table shuffle failed! " << status.get_msg();
      return status;
    }

//    // need to perform a sort to rearrange the shuffled table
//    if (!(status = local_table->Sort(0, local_table)).is_ok()) {
//      LOG(FATAL) << " table sort failed! " << status.get_msg();
//      return status;
//    }
    // use hash groupby now, because the idx rows may loose order
    group_by_fptr = PickLocalHashGroupByFptr(table->GetColumn(index_col)->GetDataType());
    // do local group by again
    if (!(status = group_by_fptr(local_table, aggregate_ops, output)).is_ok()) {
      LOG(FATAL) << "Local group by failed! " << status.get_msg();
      return status;
    }
  } else {
    output = local_table;
  }

  return Status::OK();}
}

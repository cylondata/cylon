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
#include "join.hpp"

#include <glog/logging.h>

#include <chrono>
#include <map>
#include <algorithm>
#include <vector>
#include <memory>
#include <string>

#include "arrow/compute/api.h"
#include "join_utils.hpp"
#include "../util/arrow_utils.hpp"

namespace cylon {
namespace join {

template<typename ARROW_ARRAY_TYPE, typename CPP_KEY_TYPE>
void advance(std::vector<int64_t> *subset,
             const std::shared_ptr<arrow::Int64Array> &sorted_indices,  // this is always Int64Array
             int64_t *current_index,  // always int64_t
             std::shared_ptr<arrow::Array> data_column,
             CPP_KEY_TYPE *key) {
  subset->clear();
  if (*current_index == sorted_indices->length()) {
    return;
  }
  auto data_column_casted = std::static_pointer_cast<ARROW_ARRAY_TYPE>(data_column);
  int64_t data_index = sorted_indices->Value(*current_index);
  *key = data_column_casted->GetView(data_index);
  while (*current_index < sorted_indices->length() &&
         data_column_casted->GetView(data_index) == *key) {
    subset->push_back(data_index);
    (*current_index)++;
    if (*current_index == sorted_indices->length()) {
      break;
    }
    data_index = sorted_indices->Value(*current_index);
  }
}

template<typename ARROW_ARRAY_TYPE, typename CPP_KEY_TYPE>
arrow::Status do_sorted_join(const std::shared_ptr<arrow::Table> &left_tab,
                             const std::shared_ptr<arrow::Table> &right_tab,
                             int64_t left_join_column_idx,
                             int64_t right_join_column_idx,
                             cylon::join::config::JoinType join_type,
                             std::shared_ptr<arrow::Table> *joined_table,
                             arrow::MemoryPool *memory_pool) {
  // combine chunks if multiple chunks are available
  std::shared_ptr<arrow::Table> left_tab_comb, right_tab_comb;
  arrow::Status lstatus, rstatus;
  auto t11 = std::chrono::high_resolution_clock::now();

  lstatus = cylon::join::util::CombineChunks(left_tab, left_join_column_idx,
      left_tab_comb, memory_pool);
  rstatus = cylon::join::util::CombineChunks(right_tab, right_join_column_idx,
      right_tab_comb, memory_pool);

  auto t22 = std::chrono::high_resolution_clock::now();

  if (!lstatus.ok() || !rstatus.ok()) {
    LOG(ERROR) << "Combining chunks failed!";
    return arrow::Status::Invalid("Sort join failed!");
  }

  LOG(INFO) << "Combine chunks time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t11).count();

  // sort columns
  auto left_join_column = left_tab_comb->column(left_join_column_idx)->chunk(0);
  auto right_join_column = right_tab_comb->column(right_join_column_idx)->chunk(0);

  auto t1 = std::chrono::high_resolution_clock::now();
  std::shared_ptr<arrow::Array> left_index_sorted_column;
  auto status = SortIndices(memory_pool, left_join_column, &left_index_sorted_column);
  if (status != arrow::Status::OK()) {
    LOG(FATAL) << "Failed when sorting left table to indices. " << status.ToString();
    return status;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Left sorting time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  t1 = std::chrono::high_resolution_clock::now();
  std::shared_ptr<arrow::Array> right_index_sorted_column;
  status = SortIndices(memory_pool, right_join_column, &right_index_sorted_column);
  if (status != arrow::Status::OK()) {
    LOG(FATAL) << "Failed when sorting right table to indices. " << status.ToString();
    return status;
  }
  t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "right sorting time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  CPP_KEY_TYPE left_key, right_key;
  std::vector<int64_t> left_subset, right_subset;
  int64_t left_current_index = 0;
  int64_t right_current_index = 0;

  t1 = std::chrono::high_resolution_clock::now();

  std::shared_ptr<std::vector<int64_t>> left_indices = std::make_shared<std::vector<int64_t>>();
  std::shared_ptr<std::vector<int64_t>> right_indices = std::make_shared<std::vector<int64_t>>();
  int64_t init_vec_size = std::min(left_join_column->length(), right_join_column->length());
  left_indices->reserve(init_vec_size);
  right_indices->reserve(init_vec_size);

  advance<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(&left_subset,
      std::static_pointer_cast<arrow::Int64Array>(left_index_sorted_column),
      &left_current_index,
      left_join_column,
      &left_key);

  advance<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(&right_subset,
      std::static_pointer_cast<arrow::Int64Array>(right_index_sorted_column),
      &right_current_index,
      right_join_column,
      &right_key);
  while (!left_subset.empty() && !right_subset.empty()) {
    if (left_key == right_key) {  // use a key comparator
      for (int64_t left_idx : left_subset) {
        for (int64_t right_idx : right_subset) {
          left_indices->push_back(left_idx);
          right_indices->push_back(right_idx);
        }
      }
      // advance
      advance<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(&left_subset,
                                              std::static_pointer_cast<arrow::Int64Array>(
                                                  left_index_sorted_column),
                                              &left_current_index,
                                              left_join_column,
                                              &left_key);

      advance<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(&right_subset,
                                              std::static_pointer_cast<arrow::Int64Array>(
                                                  right_index_sorted_column),
                                              &right_current_index,
                                              right_join_column,
                                              &right_key);
    } else if (left_key < right_key) {
      // if this is a left join, this is the time to include them all in the result set
      if (join_type == cylon::join::config::LEFT || join_type == cylon::join::config::FULL_OUTER) {
        for (int64_t left_idx : left_subset) {
          left_indices->push_back(left_idx);
          right_indices->push_back(-1);
        }
      }

      advance<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(&left_subset,
                                              std::static_pointer_cast<arrow::Int64Array>(
                                                  left_index_sorted_column),
                                              &left_current_index,
                                              left_join_column,
                                              &left_key);
    } else {
      // if this is a right join, this is the time to include them all in the result set
      if (join_type == cylon::join::config::RIGHT || join_type == cylon::join::config::FULL_OUTER) {
        for (int64_t right_idx : right_subset) {
          left_indices->push_back(-1);
          right_indices->push_back(right_idx);
        }
      }

      advance<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(&right_subset,
                                              std::static_pointer_cast<arrow::Int64Array>(
                                                  right_index_sorted_column),
                                              &right_current_index,
                                              right_join_column,
                                              &right_key);
    }
  }

  // specially handling left and right join
  if (join_type == cylon::join::config::LEFT || join_type == cylon::join::config::FULL_OUTER) {
    while (!left_subset.empty()) {
      for (int64_t left_idx : left_subset) {
        left_indices->push_back(left_idx);
        right_indices->push_back(-1);
      }
      advance<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(&left_subset,
                                              std::static_pointer_cast<arrow::Int64Array>(
                                                  left_index_sorted_column),
                                              &left_current_index,
                                              left_join_column,
                                              &left_key);
    }
  }

  if (join_type == cylon::join::config::RIGHT || join_type == cylon::join::config::FULL_OUTER) {
    while (!right_subset.empty()) {
      for (int64_t right_idx : right_subset) {
        left_indices->push_back(-1);
        right_indices->push_back(right_idx);
      }
      advance<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(&right_subset,
                                              std::static_pointer_cast<arrow::Int64Array>(
                                                  right_index_sorted_column),
                                              &right_current_index,
                                              right_join_column,
                                              &right_key);
    }
  }

  t2 = std::chrono::high_resolution_clock::now();

  LOG(INFO) << "Index join time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  LOG(INFO) << "Building final table with number of tuples - " << left_indices->size();

  t1 = std::chrono::high_resolution_clock::now();
  // build final table
  status = cylon::join::util::build_final_table(
      left_indices, right_indices,
      left_tab_comb,
      right_tab_comb,
      joined_table,
      memory_pool);
  t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Built final table in : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  LOG(INFO) << "Done and produced : " << left_indices->size();
  return status;
}

/**
 * Hash join impl
 * @tparam ARROW_ARRAY_TYPE type of the key array type that will be used for static casting
 * @param left_tab
 * @param right_tab
 * @param left_join_column_idx
 * @param right_join_column_idx
 * @param join_typeas
 * @param joined_table
 * @param memory_pool
 * @return arrow status
 */
template<typename ARROW_ARRAY_TYPE, typename CPP_KEY_TYPE>
arrow::Status do_hash_join(const std::shared_ptr<arrow::Table> &left_tab,
                           const std::shared_ptr<arrow::Table> &right_tab,
                           int64_t left_join_column_idx,
                           int64_t right_join_column_idx,
                           cylon::join::config::JoinType join_type,
                           std::shared_ptr<arrow::Table> *joined_table,
                           arrow::MemoryPool *memory_pool) {
  // combine chunks if multiple chunks are available
  std::shared_ptr<arrow::Table> left_tab_comb, right_tab_comb;
  arrow::Status lstatus, rstatus;
  auto t11 = std::chrono::high_resolution_clock::now();

  lstatus = cylon::join::util::CombineChunks(left_tab, left_join_column_idx,
      left_tab_comb, memory_pool);
  rstatus = cylon::join::util::CombineChunks(right_tab, right_join_column_idx,
      right_tab_comb, memory_pool);

  auto t22 = std::chrono::high_resolution_clock::now();

  if (!lstatus.ok() || !rstatus.ok()) {
    LOG(ERROR) << "Combining chunks failed!";
    return arrow::Status::Invalid("Hash join failed!");
  }

  LOG(INFO) << "Combine chunks time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t11).count();

  // sort columns
  std::shared_ptr<arrow::Array> left_idx_column = left_tab_comb->column(
      left_join_column_idx)->chunk(0);
  std::shared_ptr<arrow::Array> right_idx_column = right_tab_comb->column(
      right_join_column_idx)->chunk(0);

  std::shared_ptr<std::vector<int64_t>> left_indices = std::make_shared<std::vector<int64_t>>();
  std::shared_ptr<std::vector<int64_t>> right_indices = std::make_shared<std::vector<int64_t>>();

  int64_t init_vec_size = std::min(left_idx_column->length(), right_idx_column->length());
  left_indices->reserve(init_vec_size);
  right_indices->reserve(init_vec_size);

  auto t1 = std::chrono::high_resolution_clock::now();

  auto result = ArrowArrayIdxHashJoinKernel<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>()
      .IdxHashJoin(left_idx_column, right_idx_column, join_type, left_indices, right_indices);
//  left_indices->shrink_to_fit();
//  right_indices->shrink_to_fit();
  auto t2 = std::chrono::high_resolution_clock::now();

  if (result) {
    LOG(ERROR) << "Index join failed!";
    return arrow::Status::Invalid("Index join failed!");
  }

  LOG(INFO) << "Index join time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  LOG(INFO) << "Building final table with number of tuples - " << left_indices->size();

  t1 = std::chrono::high_resolution_clock::now();

  auto status = cylon::join::util::build_final_table(
      left_indices, right_indices,
      left_tab_comb,
      right_tab_comb,
      joined_table,
      memory_pool);

  t2 = std::chrono::high_resolution_clock::now();

  LOG(INFO) << "Built final table in : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  LOG(INFO) << "Done and produced : " << left_indices->size();

  left_indices.reset();
  right_indices.reset();

  return arrow::Status::OK();
}

template<typename ARROW_ARRAY_TYPE, typename CPP_KEY_TYPE>
arrow::Status do_join(const std::shared_ptr<arrow::Table> &left_tab,
                      const std::shared_ptr<arrow::Table> &right_tab,
                      int64_t left_join_column_idx,
                      int64_t right_join_column_idx,
                      cylon::join::config::JoinType join_type,
                      cylon::join::config::JoinAlgorithm join_algorithm,
                      std::shared_ptr<arrow::Table> *joined_table,
                      arrow::MemoryPool *memory_pool) {
  switch (join_algorithm) {
    case cylon::join::config::SORT:
      return do_sorted_join<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(left_tab,
                                                            right_tab,
                                                            left_join_column_idx,
                                                            right_join_column_idx,
                                                            join_type,
                                                            joined_table, memory_pool);
    case cylon::join::config::HASH:
      return do_hash_join<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(left_tab,
                                                          right_tab,
                                                          left_join_column_idx,
                                                          right_join_column_idx,
                                                          join_type,
                                                          joined_table, memory_pool);
  }
  return arrow::Status::OK();
}

arrow::Status joinTables(const std::vector<std::shared_ptr<arrow::Table>> &left_tabs,
                         const std::vector<std::shared_ptr<arrow::Table>> &right_tabs,
                         cylon::join::config::JoinConfig join_config,
                         std::shared_ptr<arrow::Table> *joined_table,
                         arrow::MemoryPool *memory_pool) {
  std::shared_ptr<arrow::Table> left_tab = arrow::ConcatenateTables(left_tabs,
      arrow::ConcatenateTablesOptions::Defaults(),
      memory_pool).ValueOrDie();
  std::shared_ptr<arrow::Table> right_tab = arrow::ConcatenateTables(right_tabs,
      arrow::ConcatenateTablesOptions::Defaults(),
      memory_pool).ValueOrDie();
  std::shared_ptr<arrow::Table> left_tab_combined;
  std::shared_ptr<arrow::Table> right_tab_combined;

  arrow::Status left_combine_stat = left_tab->CombineChunks(memory_pool, &left_tab_combined);
  if (left_combine_stat != arrow::Status::OK()) {
    LOG(FATAL) << "Error in combining table chunks of left table." << left_combine_stat.ToString();
    return left_combine_stat;
  }
  if (left_tabs.size() > 1) {
    for (const auto &t : left_tabs) {
      arrow::Status status = cylon::util::free_table(t);
      if (status != arrow::Status::OK()) {
        LOG(FATAL) << "Failed to free table" << status.ToString();
        return status;
      }
    }
  }

  arrow::Status right_combine_stat = right_tab->CombineChunks(memory_pool, &right_tab_combined);
  if (right_combine_stat != arrow::Status::OK()) {
    LOG(FATAL) << "Error in combining table chunks of right table." << left_combine_stat.ToString();
    return right_combine_stat;
  }

  if (right_tabs.size() > 1) {
    for (const auto& t : right_tabs) {
      arrow::Status status = cylon::util::free_table(t);
      if (status != arrow::Status::OK()) {
        LOG(FATAL) << "Failed to free table" << status.ToString();
        return status;
      }
    }
  }

  return cylon::join::joinTables(left_tab_combined,
                                 right_tab_combined,
                                 join_config,
                                 joined_table,
                                 memory_pool);
}

arrow::Status joinTables(const std::shared_ptr<arrow::Table> &left_tab,
                         const std::shared_ptr<arrow::Table> &right_tab,
                         cylon::join::config::JoinConfig join_config,
                         std::shared_ptr<arrow::Table> *joined_table,
                         arrow::MemoryPool *memory_pool) {
  auto left_type = left_tab->column(join_config.GetLeftColumnIdx())->type()->id();
  auto right_type = right_tab->column(join_config.GetRightColumnIdx())->type()->id();

  if (left_type != right_type) {
    LOG(FATAL) << "The join column types of two tables mismatches.";
    return arrow::Status::Invalid<std::string>("The join column types of two tables mismatches.");
  }

  switch (left_type) {
    case arrow::Type::NA:break;
    case arrow::Type::BOOL:break;
    case arrow::Type::UINT8:
      return do_join<arrow::NumericArray<arrow::UInt8Type>, uint8_t>(left_tab,
                                                                   right_tab,
                                                                   join_config.GetLeftColumnIdx(),
                                                                   join_config.GetRightColumnIdx(),
                                                                   join_config.GetType(),
                                                                   join_config.GetAlgorithm(),
                                                                   joined_table,
                                                                   memory_pool);
    case arrow::Type::INT8:
      return do_join<arrow::NumericArray<arrow::Int8Type>, int8_t>(left_tab,
                                                                   right_tab,
                                                                   join_config.GetLeftColumnIdx(),
                                                                   join_config.GetRightColumnIdx(),
                                                                   join_config.GetType(),
                                                                   join_config.GetAlgorithm(),
                                                                   joined_table,
                                                                   memory_pool);
    case arrow::Type::UINT16:
      return do_join<arrow::NumericArray<arrow::UInt16Type>, uint16_t>(left_tab,
                                                                   right_tab,
                                                                   join_config.GetLeftColumnIdx(),
                                                                   join_config.GetRightColumnIdx(),
                                                                   join_config.GetType(),
                                                                   join_config.GetAlgorithm(),
                                                                   joined_table,
                                                                   memory_pool);
    case arrow::Type::INT16:
      return do_join<arrow::NumericArray<arrow::Int16Type>, int16_t>(left_tab,
                                                                   right_tab,
                                                                   join_config.GetLeftColumnIdx(),
                                                                   join_config.GetRightColumnIdx(),
                                                                   join_config.GetType(),
                                                                   join_config.GetAlgorithm(),
                                                                   joined_table,
                                                                   memory_pool);
    case arrow::Type::UINT32:
      return do_join<arrow::NumericArray<arrow::UInt32Type>, uint32_t>(left_tab,
                                                                   right_tab,
                                                                   join_config.GetLeftColumnIdx(),
                                                                   join_config.GetRightColumnIdx(),
                                                                   join_config.GetType(),
                                                                   join_config.GetAlgorithm(),
                                                                   joined_table,
                                                                   memory_pool);
    case arrow::Type::INT32:
      return do_join<arrow::NumericArray<arrow::Int32Type>, int32_t>(left_tab,
                                                                   right_tab,
                                                                   join_config.GetLeftColumnIdx(),
                                                                   join_config.GetRightColumnIdx(),
                                                                   join_config.GetType(),
                                                                   join_config.GetAlgorithm(),
                                                                   joined_table,
                                                                   memory_pool);
    case arrow::Type::UINT64:
      return do_join<arrow::NumericArray<arrow::UInt64Type>, uint64_t>(left_tab,
                                                                   right_tab,
                                                                   join_config.GetLeftColumnIdx(),
                                                                   join_config.GetRightColumnIdx(),
                                                                   join_config.GetType(),
                                                                   join_config.GetAlgorithm(),
                                                                   joined_table,
                                                                   memory_pool);
    case arrow::Type::INT64:
      return do_join<arrow::NumericArray<arrow::Int64Type>, int64_t>(left_tab,
                                                                   right_tab,
                                                                   join_config.GetLeftColumnIdx(),
                                                                   join_config.GetRightColumnIdx(),
                                                                   join_config.GetType(),
                                                                   join_config.GetAlgorithm(),
                                                                   joined_table,
                                                                   memory_pool);;
    case arrow::Type::HALF_FLOAT:
      return do_join<arrow::NumericArray<arrow::HalfFloatType>, float_t>(left_tab,
                                                                   right_tab,
                                                                   join_config.GetLeftColumnIdx(),
                                                                   join_config.GetRightColumnIdx(),
                                                                   join_config.GetType(),
                                                                   join_config.GetAlgorithm(),
                                                                   joined_table,
                                                                   memory_pool);
    case arrow::Type::FLOAT:
      return do_join<arrow::NumericArray<arrow::FloatType>, float_t>(left_tab,
                                                                   right_tab,
                                                                   join_config.GetLeftColumnIdx(),
                                                                   join_config.GetRightColumnIdx(),
                                                                   join_config.GetType(),
                                                                   join_config.GetAlgorithm(),
                                                                   joined_table,
                                                                   memory_pool);
    case arrow::Type::DOUBLE:
      return do_join<arrow::NumericArray<arrow::DoubleType>, double_t>(left_tab,
                                                                   right_tab,
                                                                   join_config.GetLeftColumnIdx(),
                                                                   join_config.GetRightColumnIdx(),
                                                                   join_config.GetType(),
                                                                   join_config.GetAlgorithm(),
                                                                   joined_table,
                                                                   memory_pool);
    case arrow::Type::STRING:
      return do_join<arrow::StringArray, arrow::util::string_view>(left_tab,
                                                                   right_tab,
                                                                   join_config.GetLeftColumnIdx(),
                                                                   join_config.GetRightColumnIdx(),
                                                                   join_config.GetType(),
                                                                   join_config.GetAlgorithm(),
                                                                   joined_table,
                                                                   memory_pool);
    case arrow::Type::BINARY:
      return do_join<arrow::BinaryArray, arrow::util::string_view>(left_tab,
                                                                   right_tab,
                                                                   join_config.GetLeftColumnIdx(),
                                                                   join_config.GetRightColumnIdx(),
                                                                   join_config.GetType(),
                                                                   join_config.GetAlgorithm(),
                                                                   joined_table,
                                                                   memory_pool);
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
  return arrow::Status::OK();
}
}  // namespace join
}  // namespace cylon

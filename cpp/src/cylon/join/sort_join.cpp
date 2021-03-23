#include "sort_join.hpp"

#include "../arrow/arrow_comparator.hpp"
#include "../arrow/arrow_kernels.hpp"
#include "arrow/api.h"

namespace cylon {
namespace join {

/* SINGLE INDEX */

template<typename ARROW_ARRAY_TYPE, typename CPP_KEY_TYPE>
inline void advance(
    std::vector<int64_t> *subset,
    const std::shared_ptr<arrow::UInt64Array> &sorted_indices,  // this is always UInt64Array
    int64_t *current_index,                                     // always int64_t
    std::shared_ptr<arrow::Array> data_column, CPP_KEY_TYPE *key) {
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

template<typename ARROW_T, typename CPP_KEY_TYPE>
static inline arrow::Status do_sorted_join(
    const std::shared_ptr<arrow::Table> &left_tab, const std::shared_ptr<arrow::Table> &right_tab,
    int64_t left_join_column_idx, int64_t right_join_column_idx,
    cylon::join::config::JoinType join_type, const std::string &left_table_prefix,
    const std::string &right_table_prefix, std::shared_ptr<arrow::Table> *joined_table,
    arrow::MemoryPool *memory_pool) {
  using ARROW_ARRAY_TYPE = typename arrow::TypeTraits<ARROW_T>::ArrayType;

  // combine chunks if multiple chunks are available
  std::shared_ptr<arrow::Table> left_tab_comb, right_tab_comb;
  arrow::Status lstatus, rstatus;
  auto t11 = std::chrono::high_resolution_clock::now();

  lstatus =
      cylon::join::util::CombineChunks(left_tab, left_join_column_idx, left_tab_comb, memory_pool);
  rstatus = cylon::join::util::CombineChunks(right_tab, right_join_column_idx, right_tab_comb,
                                             memory_pool);

  auto t22 = std::chrono::high_resolution_clock::now();

  if (!lstatus.ok() || !rstatus.ok()) {
    LOG(ERROR) << "Combining chunks failed!";
    return arrow::Status::Invalid("Sort join failed!");
  }

  LOG(INFO) << "Combine chunks time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t11).count();

  // sort columns
  auto left_join_column =
      cylon::util::GetChunkOrEmptyArray(left_tab_comb->column(left_join_column_idx), 0);
  auto right_join_column =
      cylon::util::GetChunkOrEmptyArray(right_tab_comb->column(right_join_column_idx), 0);

  auto t1 = std::chrono::high_resolution_clock::now();
  std::shared_ptr<arrow::UInt64Array> left_index_sorted_column;
  auto status = SortIndices(memory_pool, left_join_column, left_index_sorted_column);
  if (!status.ok()) {
    LOG(FATAL) << "Failed when sorting left table to indices. " << status.ToString();
    return status;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Left sorting time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  t1 = std::chrono::high_resolution_clock::now();
  std::shared_ptr<arrow::UInt64Array> right_index_sorted_column;
  status = SortIndices(memory_pool, right_join_column, right_index_sorted_column);
  if (!status.ok()) {
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

  std::vector<int64_t> left_indices, right_indices;
  int64_t init_vec_size = std::min(left_join_column->length(), right_join_column->length());
  left_indices.reserve(init_vec_size);
  right_indices.reserve(init_vec_size);

  advance<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(&left_subset, left_index_sorted_column,
                                          &left_current_index, left_join_column, &left_key);

  advance<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(&right_subset, right_index_sorted_column,
                                          &right_current_index, right_join_column, &right_key);
  while (!left_subset.empty() && !right_subset.empty()) {
    if (left_key == right_key) {  // use a key comparator
      for (int64_t left_idx : left_subset) {
        for (int64_t right_idx : right_subset) {
          left_indices.push_back(left_idx);
          right_indices.push_back(right_idx);
        }
      }
      // advance
      advance<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(&left_subset, left_index_sorted_column,
                                              &left_current_index, left_join_column, &left_key);

      advance<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(&right_subset, right_index_sorted_column,
                                              &right_current_index, right_join_column, &right_key);
    } else if (left_key < right_key) {
      // if this is a left join, this is the time to include them all in the result set
      if (join_type == cylon::join::config::LEFT || join_type == cylon::join::config::FULL_OUTER) {
        for (int64_t left_idx : left_subset) {
          left_indices.push_back(left_idx);
          right_indices.push_back(-1);
        }
      }

      advance<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(&left_subset, left_index_sorted_column,
                                              &left_current_index, left_join_column, &left_key);
    } else {
      // if this is a right join, this is the time to include them all in the result set
      if (join_type == cylon::join::config::RIGHT || join_type == cylon::join::config::FULL_OUTER) {
        for (int64_t right_idx : right_subset) {
          left_indices.push_back(-1);
          right_indices.push_back(right_idx);
        }
      }

      advance<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(&right_subset, right_index_sorted_column,
                                              &right_current_index, right_join_column, &right_key);
    }
  }

  // specially handling left and right join
  if (join_type == cylon::join::config::LEFT || join_type == cylon::join::config::FULL_OUTER) {
    while (!left_subset.empty()) {
      for (int64_t left_idx : left_subset) {
        left_indices.push_back(left_idx);
        right_indices.push_back(-1);
      }
      advance<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(&left_subset, left_index_sorted_column,
                                              &left_current_index, left_join_column, &left_key);
    }
  }

  if (join_type == cylon::join::config::RIGHT || join_type == cylon::join::config::FULL_OUTER) {
    while (!right_subset.empty()) {
      for (int64_t right_idx : right_subset) {
        left_indices.push_back(-1);
        right_indices.push_back(right_idx);
      }
      advance<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(&right_subset, right_index_sorted_column,
                                              &right_current_index, right_join_column, &right_key);
    }
  }

  // clear the sort columns
  left_index_sorted_column.reset();
  right_index_sorted_column.reset();
  t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Index join time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  LOG(INFO) << "Building final table with number of tuples - " << left_indices.size();

  t1 = std::chrono::high_resolution_clock::now();
  // build final table
  status = cylon::join::util::build_final_table(left_indices, right_indices, left_tab_comb,
                                                right_tab_comb, left_table_prefix,
                                                right_table_prefix, joined_table, memory_pool);
  t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Built final table in : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  LOG(INFO) << "Done and produced : " << left_indices.size();
  return status;
}

/*  SINGLE INDEX INPLACE */

template<typename ARROW_ARRAY_TYPE, typename CPP_KEY_TYPE>
inline void advance_inplace_array(std::vector<int64_t> *subset,
                                  int64_t *current_index,  // always int64_t
                                  std::shared_ptr<arrow::Array> data_column, int64_t length,
                                  CPP_KEY_TYPE *key) {
  subset->clear();
  if (*current_index == length) {
    return;
  }
  auto data_column_casted = std::static_pointer_cast<ARROW_ARRAY_TYPE>(data_column);
  *key = data_column_casted->GetView(*current_index);
  while (*current_index < length && data_column_casted->GetView(*current_index) == *key) {
    subset->push_back(*current_index);
    (*current_index)++;
    if (*current_index == length) {
      break;
    }
  }
}

template<typename ARROW_T, typename CPP_KEY_TYPE>
arrow::Status do_inplace_sorted_join(const std::shared_ptr<arrow::Table> &left_tab,
                                     const std::shared_ptr<arrow::Table> &right_tab,
                                     int64_t left_join_column_idx, int64_t right_join_column_idx,
                                     cylon::join::config::JoinType join_type,
                                     const std::string &left_table_prefix,
                                     const std::string &right_table_prefix,
                                     std::shared_ptr<arrow::Table> *joined_table,
                                     arrow::MemoryPool *memory_pool) {
  using ARROW_ARRAY_TYPE = typename arrow::TypeTraits<ARROW_T>::ArrayType;
  // combine chunks if multiple chunks are available
  std::shared_ptr<arrow::Table> left_tab_comb, right_tab_comb;
  arrow::Status lstatus, rstatus;
  auto t11 = std::chrono::high_resolution_clock::now();

  lstatus =
      cylon::join::util::CombineChunks(left_tab, left_join_column_idx, left_tab_comb, memory_pool);
  rstatus = cylon::join::util::CombineChunks(right_tab, right_join_column_idx, right_tab_comb,
                                             memory_pool);

  auto t22 = std::chrono::high_resolution_clock::now();

  if (!lstatus.ok() || !rstatus.ok()) {
    LOG(ERROR) << "Combining chunks failed!";
    return arrow::Status::Invalid("Sort join failed!");
  }
  LOG(INFO) << "Combine chunks time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t11).count();

  // sort columns
  auto left_join_column =
      cylon::util::GetChunkOrEmptyArray(left_tab_comb->column(left_join_column_idx), 0);
  auto right_join_column =
      cylon::util::GetChunkOrEmptyArray(right_tab_comb->column(right_join_column_idx), 0);

  auto t1 = std::chrono::high_resolution_clock::now();
  std::shared_ptr<arrow::UInt64Array> left_index_sorted_column;
  auto status = SortIndicesInPlace(memory_pool, left_join_column, left_index_sorted_column);
  if (!status.ok()) {
    LOG(FATAL) << "Failed when sorting left table to indices. " << status.ToString();
    return status;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Left sorting time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  t1 = std::chrono::high_resolution_clock::now();
  std::shared_ptr<arrow::UInt64Array> right_index_sorted_column;
  status = SortIndicesInPlace(memory_pool, right_join_column, right_index_sorted_column);
  if (!status.ok()) {
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

  std::vector<int64_t> left_indices, right_indices;
  int64_t init_vec_size = std::min(left_join_column->length(), right_join_column->length());
  left_indices.reserve(init_vec_size);
  right_indices.reserve(init_vec_size);

  int64_t col_length = left_join_column->length();
  int64_t right_col_length = right_join_column->length();

  advance_inplace_array<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(&left_subset, &left_current_index,
                                                        left_join_column, col_length, &left_key);

  advance_inplace_array<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(
      &right_subset, &right_current_index, right_join_column, right_col_length, &right_key);
  while (!left_subset.empty() && !right_subset.empty()) {
    if (left_key == right_key) {  // use a key comparator
      for (int64_t left_idx : left_subset) {
        for (int64_t right_idx : right_subset) {
          left_indices.push_back(left_idx);
          right_indices.push_back(right_idx);
        }
      }
      // advance
      advance_inplace_array<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(
          &left_subset, &left_current_index, left_join_column, col_length, &left_key);

      advance_inplace_array<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(
          &right_subset, &right_current_index, right_join_column, right_col_length, &right_key);
    } else if (left_key < right_key) {
      // if this is a left join, this is the time to include them all in the result set
      if (join_type == cylon::join::config::LEFT || join_type == cylon::join::config::FULL_OUTER) {
        for (int64_t left_idx : left_subset) {
          left_indices.push_back(left_idx);
          right_indices.push_back(-1);
        }
      }

      advance_inplace_array<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(
          &left_subset, &left_current_index, left_join_column, col_length, &left_key);
    } else {
      // if this is a right join, this is the time to include them all in the result set
      if (join_type == cylon::join::config::RIGHT || join_type == cylon::join::config::FULL_OUTER) {
        for (int64_t right_idx : right_subset) {
          left_indices.push_back(-1);
          right_indices.push_back(right_idx);
        }
      }

      advance_inplace_array<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(
          &right_subset, &right_current_index, right_join_column, right_col_length, &right_key);
    }
  }

  // specially handling left and right join
  if (join_type == cylon::join::config::LEFT || join_type == cylon::join::config::FULL_OUTER) {
    while (!left_subset.empty()) {
      for (int64_t left_idx : left_subset) {
        left_indices.push_back(left_idx);
        right_indices.push_back(-1);
      }
      advance_inplace_array<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(
          &left_subset, &left_current_index, left_join_column, col_length, &left_key);
    }
  }

  if (join_type == cylon::join::config::RIGHT || join_type == cylon::join::config::FULL_OUTER) {
    while (!right_subset.empty()) {
      for (int64_t right_idx : right_subset) {
        left_indices.push_back(-1);
        right_indices.push_back(right_idx);
      }
      advance_inplace_array<ARROW_ARRAY_TYPE, CPP_KEY_TYPE>(
          &right_subset, &right_current_index, right_join_column, right_col_length, &right_key);
    }
  }

  t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Index join time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  LOG(INFO) << "Building final table with number of tuples - " << left_indices.size();

  t1 = std::chrono::high_resolution_clock::now();
  // build final table
  status = cylon::join::util::build_final_table_inplace_index(
      left_join_column_idx, right_join_column_idx, left_indices, right_indices,
      left_index_sorted_column, right_index_sorted_column, left_tab_comb, right_tab_comb,
      left_table_prefix, right_table_prefix, joined_table, memory_pool);
  t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Built final table in : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  LOG(INFO) << "Done and produced : " << left_indices.size();

  return status;
}

/*  MULTI INDEX */

inline void advance_multi_index(
    std::vector<int64_t> *subset,
    const std::shared_ptr<arrow::UInt64Array> &sorted_indices,  // this is always UInt64Array
    int64_t *current_index,                                     // always int64_t
    cylon::TableRowIndexEqualTo *comp, int64_t *key_index) {
  subset->clear();
  if (*current_index == sorted_indices->length()) {
    return;
  }

  int64_t data_index = sorted_indices->Value(*current_index);
  *key_index = data_index;
  while (*current_index < sorted_indices->length() && comp->compare(data_index, *key_index) == 0) {
    subset->push_back(data_index);
    (*current_index)++;
    if (*current_index == sorted_indices->length()) {
      break;
    }
    data_index = sorted_indices->Value(*current_index);
  }
}

static inline arrow::Status do_multi_index_sorted_join(
    const std::shared_ptr<arrow::Table> &left_tab, const std::shared_ptr<arrow::Table> &right_tab,
    const std::vector<int32_t> &left_join_column_indices,
    const std::vector<int32_t> &right_join_column_indices, cylon::join::config::JoinType join_type,
    const std::string &left_table_prefix, const std::string &right_table_prefix,
    std::shared_ptr<arrow::Table> *joined_table, arrow::MemoryPool *memory_pool) {
  // combine chunks if multiple chunks are available
  arrow::Result<std::shared_ptr<arrow::Table>> left_status, right_status;
  auto t11 = std::chrono::high_resolution_clock::now();

  left_status = left_tab->CombineChunks(memory_pool);
  right_status = right_tab->CombineChunks(memory_pool);

  auto t22 = std::chrono::high_resolution_clock::now();

  if (!left_status.ok() || !right_status.ok()) {
    LOG(ERROR) << "Combining chunks failed!";
    return arrow::Status::Invalid("Sort join failed!");
  }
  const std::shared_ptr<arrow::Table> left_tab_comb = left_status.ValueOrDie();
  const std::shared_ptr<arrow::Table> right_tab_comb = right_status.ValueOrDie();

  LOG(INFO) << "Combine chunks time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t11).count();

  // create sorter and do index sort
  auto t1 = std::chrono::high_resolution_clock::now();

  std::shared_ptr<arrow::UInt64Array> left_index_sorted_column;
  auto status = cylon::SortIndicesMultiColumns(memory_pool, left_tab_comb, left_join_column_indices,
                                               left_index_sorted_column);
  if (!status.ok()) {
    LOG(FATAL) << "Failed when sorting left table to indices. " << status.ToString();
    return status;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Left sorting time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  std::shared_ptr<arrow::UInt64Array> right_index_sorted_column;
  status = cylon::SortIndicesMultiColumns(memory_pool, right_tab_comb, right_join_column_indices,
                                          right_index_sorted_column);
  if (!status.ok()) {
    LOG(FATAL) << "Failed when sorting right table to indices. " << status.ToString();
    return status;
  }

  t1 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "right sorting time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t2).count();

  // create comparator
  auto left_tab_index_comparator =
      cylon::TableRowIndexEqualTo(left_tab_comb, left_join_column_indices);
  auto right_tab_index_comparator =
      cylon::TableRowIndexEqualTo(right_tab_comb, right_join_column_indices);

  auto mult_tab_comparator = cylon::TwoTableRowIndexEqualTo(left_tab_comb,
                                                            right_tab_comb,
                                                            left_join_column_indices,
                                                            right_join_column_indices);

  int64_t left_key_index = 0, right_key_index = 0;  // reference indices
  std::vector<int64_t> left_subset, right_subset;

  int64_t left_current_index = 0;
  int64_t right_current_index = 0;

  std::vector<int64_t> left_indices, right_indices;
  int64_t init_vec_size = std::min(left_tab->num_rows(), right_tab->num_rows());
  left_indices.reserve(init_vec_size);
  right_indices.reserve(init_vec_size);

  advance_multi_index(&left_subset, left_index_sorted_column, &left_current_index,
                      &left_tab_index_comparator, &left_key_index);

  advance_multi_index(&right_subset, right_index_sorted_column, &right_current_index,
                      &right_tab_index_comparator, &right_key_index);

  while (!left_subset.empty() && !right_subset.empty()) {
    auto com = mult_tab_comparator.compare(0, left_key_index, 1, right_key_index);
    if (com == 0) {  // use a key comparator
      for (int64_t left_idx : left_subset) {
        for (int64_t right_idx : right_subset) {
          left_indices.push_back(left_idx);
          right_indices.push_back(right_idx);
        }
      }
      // advance
      advance_multi_index(&left_subset, left_index_sorted_column, &left_current_index,
                          &left_tab_index_comparator, &left_key_index);

      advance_multi_index(&right_subset, right_index_sorted_column, &right_current_index,
                          &right_tab_index_comparator, &right_key_index);
    } else if (com < 0) {
      // if this is a left join, this is the time to include them all in the result set
      if (join_type == cylon::join::config::LEFT || join_type == cylon::join::config::FULL_OUTER) {
        for (int64_t left_idx : left_subset) {
          left_indices.push_back(left_idx);
          right_indices.push_back(-1);
        }
      }

      advance_multi_index(&left_subset, left_index_sorted_column, &left_current_index,
                          &left_tab_index_comparator, &left_key_index);
    } else {
      // if this is a right join, this is the time to include them all in the result set
      if (join_type == cylon::join::config::RIGHT || join_type == cylon::join::config::FULL_OUTER) {
        for (int64_t right_idx : right_subset) {
          left_indices.push_back(-1);
          right_indices.push_back(right_idx);
        }
      }

      advance_multi_index(&right_subset, right_index_sorted_column, &right_current_index,
                          &right_tab_index_comparator, &right_key_index);
    }
  }

  // specially handling left and right join
  if (join_type == cylon::join::config::LEFT || join_type == cylon::join::config::FULL_OUTER) {
    while (!left_subset.empty()) {
      for (int64_t left_idx : left_subset) {
        left_indices.push_back(left_idx);
        right_indices.push_back(-1);
      }
      advance_multi_index(&left_subset, left_index_sorted_column, &left_current_index,
                          &left_tab_index_comparator, &left_key_index);
    }
  }

  if (join_type == cylon::join::config::RIGHT || join_type == cylon::join::config::FULL_OUTER) {
    while (!right_subset.empty()) {
      for (int64_t right_idx : right_subset) {
        left_indices.push_back(-1);
        right_indices.push_back(right_idx);
      }
      advance_multi_index(&right_subset, right_index_sorted_column, &right_current_index,
                          &right_tab_index_comparator, &right_key_index);
    }
  }

  // clear the sort columns
  left_index_sorted_column.reset();
  right_index_sorted_column.reset();
  t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Index join time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  LOG(INFO) << "Building final table with number of tuples - " << left_indices.size();

  t1 = std::chrono::high_resolution_clock::now();
  // build final table
  status = cylon::join::util::build_final_table(left_indices, right_indices, left_tab_comb,
                                                right_tab_comb, left_table_prefix,
                                                right_table_prefix, joined_table, memory_pool);
  t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Built final table in : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  LOG(INFO) << "Done and produced : " << left_indices.size();
  return status;
}

template<typename ARROW_T, typename CPP_KEY_TYPE = typename ARROW_T::c_type>
arrow::Status do_single_column_join(const std::shared_ptr<arrow::Table> &left_tab,
                                    const std::shared_ptr<arrow::Table> &right_tab, int64_t left_join_column_idx,
                                    int64_t right_join_column_idx, cylon::join::config::JoinType join_type,
                                    cylon::join::config::JoinAlgorithm join_algorithm,
                                    const std::string left_table_prefix, const std::string right_table_prefix,
                                    std::shared_ptr<arrow::Table> *joined_table, arrow::MemoryPool *memory_pool) {
  if (arrow::is_number_type<ARROW_T>::value) {
    return do_inplace_sorted_join<ARROW_T, CPP_KEY_TYPE>(
        left_tab, right_tab, left_join_column_idx, right_join_column_idx, join_type,
        left_table_prefix, right_table_prefix, joined_table, memory_pool);
  } else {
    return do_sorted_join<ARROW_T, CPP_KEY_TYPE>(left_tab, right_tab, left_join_column_idx, right_join_column_idx,
                                                 join_type, left_table_prefix, right_table_prefix, joined_table,
                                                 memory_pool);
  }
}

arrow::Status SortJoin(const std::shared_ptr<arrow::Table> &left_tab,
                       const std::shared_ptr<arrow::Table> &right_tab,
                       const config::JoinConfig &join_config,
                       std::shared_ptr<arrow::Table> *joined_table,
                       arrow::MemoryPool *memory_pool) {
  auto left_indices = join_config.GetLeftColumnIdx();
  auto right_indices = join_config.GetRightColumnIdx();

  // sort joins
  if (left_indices.size() == 1 && right_indices.size() == 1) {
    const auto &left_type = left_tab->column(left_indices[0])->type()->id();
    const auto &right_type = right_tab->column(right_indices[0])->type()->id();

    if (left_type != right_type) {
      LOG(FATAL) << "The join column types of two tables mismatches.";
      return arrow::Status::Invalid<std::string>("The join column types of two tables mismatches.");
    }

    switch (left_type) {
      case arrow::Type::NA:break;
      case arrow::Type::BOOL:break;
      case arrow::Type::UINT8:
        return do_single_column_join<arrow::UInt8Type>(
            left_tab, right_tab, left_indices[0], right_indices[0], join_config.GetType(),
            join_config.GetAlgorithm(), join_config.GetLeftTableSuffix(),
            join_config.GetRightTableSuffix(), joined_table, memory_pool);
      case arrow::Type::INT8:
        return do_single_column_join<arrow::Int8Type>(
            left_tab, right_tab, left_indices[0], right_indices[0], join_config.GetType(),
            join_config.GetAlgorithm(), join_config.GetLeftTableSuffix(),
            join_config.GetRightTableSuffix(), joined_table, memory_pool);
      case arrow::Type::UINT16:
        return do_single_column_join<arrow::UInt16Type>(
            left_tab, right_tab, left_indices[0], right_indices[0], join_config.GetType(),
            join_config.GetAlgorithm(), join_config.GetLeftTableSuffix(),
            join_config.GetRightTableSuffix(), joined_table, memory_pool);
      case arrow::Type::INT16:
        return do_single_column_join<arrow::Int16Type>(
            left_tab, right_tab, left_indices[0], right_indices[0], join_config.GetType(),
            join_config.GetAlgorithm(), join_config.GetLeftTableSuffix(),
            join_config.GetRightTableSuffix(), joined_table, memory_pool);
      case arrow::Type::UINT32:
        return do_single_column_join<arrow::UInt32Type>(
            left_tab, right_tab, left_indices[0], right_indices[0], join_config.GetType(),
            join_config.GetAlgorithm(), join_config.GetLeftTableSuffix(),
            join_config.GetRightTableSuffix(), joined_table, memory_pool);
      case arrow::Type::INT32:
        return do_single_column_join<arrow::Int32Type>(
            left_tab, right_tab, left_indices[0], right_indices[0], join_config.GetType(),
            join_config.GetAlgorithm(), join_config.GetLeftTableSuffix(),
            join_config.GetRightTableSuffix(), joined_table, memory_pool);
      case arrow::Type::UINT64:
        return do_single_column_join<arrow::UInt64Type>(
            left_tab, right_tab, left_indices[0], right_indices[0], join_config.GetType(),
            join_config.GetAlgorithm(), join_config.GetLeftTableSuffix(),
            join_config.GetRightTableSuffix(), joined_table, memory_pool);
      case arrow::Type::INT64:
        return do_single_column_join<arrow::Int64Type>(
            left_tab, right_tab, left_indices[0], right_indices[0], join_config.GetType(),
            join_config.GetAlgorithm(), join_config.GetLeftTableSuffix(),
            join_config.GetRightTableSuffix(), joined_table, memory_pool);;
      case arrow::Type::HALF_FLOAT:
        return do_single_column_join<arrow::HalfFloatType>(
            left_tab, right_tab, left_indices[0], right_indices[0], join_config.GetType(),
            join_config.GetAlgorithm(), join_config.GetLeftTableSuffix(),
            join_config.GetRightTableSuffix(), joined_table, memory_pool);
      case arrow::Type::FLOAT:
        return do_single_column_join<arrow::FloatType>(
            left_tab, right_tab, left_indices[0], right_indices[0], join_config.GetType(),
            join_config.GetAlgorithm(), join_config.GetLeftTableSuffix(),
            join_config.GetRightTableSuffix(), joined_table, memory_pool);
      case arrow::Type::DOUBLE:
        return do_single_column_join<arrow::DoubleType>(
            left_tab, right_tab, left_indices[0], right_indices[0], join_config.GetType(),
            join_config.GetAlgorithm(), join_config.GetLeftTableSuffix(),
            join_config.GetRightTableSuffix(), joined_table, memory_pool);
      case arrow::Type::STRING:
        return do_single_column_join<arrow::StringType, arrow::util::string_view>(
            left_tab, right_tab, left_indices[0], right_indices[0], join_config.GetType(),
            join_config.GetAlgorithm(), join_config.GetLeftTableSuffix(),
            join_config.GetRightTableSuffix(), joined_table, memory_pool);
      case arrow::Type::BINARY:
        return do_single_column_join<arrow::BinaryType, arrow::util::string_view>(
            left_tab, right_tab, left_indices[0], right_indices[0], join_config.GetType(),
            join_config.GetAlgorithm(), join_config.GetLeftTableSuffix(),
            join_config.GetRightTableSuffix(), joined_table, memory_pool);
      case arrow::Type::FIXED_SIZE_BINARY:
        return do_single_column_join<arrow::FixedSizeBinaryType, arrow::util::string_view>(
            left_tab, right_tab, left_indices[0], right_indices[0], join_config.GetType(),
            join_config.GetAlgorithm(), join_config.GetLeftTableSuffix(),
            join_config.GetRightTableSuffix(), joined_table, memory_pool);
      case arrow::Type::DATE32:
        return do_single_column_join<arrow::Date32Type>(
            left_tab, right_tab, left_indices[0], right_indices[0], join_config.GetType(),
            join_config.GetAlgorithm(), join_config.GetLeftTableSuffix(),
            join_config.GetRightTableSuffix(), joined_table, memory_pool);
      case arrow::Type::DATE64:
        return do_single_column_join<arrow::Date64Type>(
            left_tab, right_tab, left_indices[0], right_indices[0], join_config.GetType(),
            join_config.GetAlgorithm(), join_config.GetLeftTableSuffix(),
            join_config.GetRightTableSuffix(), joined_table, memory_pool);
      case arrow::Type::TIMESTAMP:
        return do_single_column_join<arrow::TimestampType>(
            left_tab, right_tab, left_indices[0], right_indices[0], join_config.GetType(),
            join_config.GetAlgorithm(), join_config.GetLeftTableSuffix(),
            join_config.GetRightTableSuffix(), joined_table, memory_pool);
      case arrow::Type::TIME32:
        return do_single_column_join<arrow::Time32Type>(
            left_tab, right_tab, left_indices[0], right_indices[0], join_config.GetType(),
            join_config.GetAlgorithm(), join_config.GetLeftTableSuffix(),
            join_config.GetRightTableSuffix(), joined_table, memory_pool);
      case arrow::Type::TIME64:
        return do_single_column_join<arrow::Time64Type>(
            left_tab, right_tab, left_indices[0], right_indices[0], join_config.GetType(),
            join_config.GetAlgorithm(), join_config.GetLeftTableSuffix(),
            join_config.GetRightTableSuffix(), joined_table, memory_pool);
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
    return arrow::Status::Invalid("Un-supported type");
  } else {
    return do_multi_index_sorted_join(left_tab,
                                      right_tab,
                                      left_indices,
                                      right_indices,
                                      join_config.GetType(),
                                      join_config.GetLeftTableSuffix(),
                                      join_config.GetRightTableSuffix(),
                                      joined_table,
                                      memory_pool);
  }
}
}  // namespace join
}  // namespace cylon
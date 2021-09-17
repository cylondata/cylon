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

#ifndef CYLON_SRC_CYLON_INDEXING_INDEX_H_
#define CYLON_SRC_CYLON_INDEXING_INDEX_H_

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/visitor_inline.h>

#include "cylon/status.hpp"

namespace cylon {

class Table;

enum IndexingType {
  Range = 0,
  Linear = 1,
  Hash = 2,
  BinaryTree = 3,
  BTree = 4,
};

class BaseArrowIndex {
 public:
  static constexpr int kNoColumnId = -1;

  BaseArrowIndex(int col_id, int64_t size, arrow::MemoryPool *pool)
      : size_(size), col_id_(col_id), pool_(pool) {};

  virtual ~BaseArrowIndex() = default;

  /**
   * Find all locations of the search_param
   * @param search_param
   * @param find_index
   * @return
   */
  virtual Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
                                 std::shared_ptr<arrow::Int64Array> *locations) = 0;
  /**
   * Find the first position of search_param
   * @param search_param
   * @param find_index
   * @return
   */
  virtual Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
                                 int64_t *find_index) = 0;

  /**
   * Find all locations of the values in search_param array
   * @param search_param
   * @param filter_location
   * @return
   */
  virtual Status LocationByVector(const std::shared_ptr<arrow::Array> &search_param,
                                  std::shared_ptr<arrow::Int64Array> *locations) = 0;

  /**
   * Finds the index range between start_value and end_value. If range can not be determined,
   * `KeyError` Status will be returned, and values of start_index and end_index are undefined
   * ex: index = [1, 2, 2, 3, 4, 4, 5], start_value = 2, end_value = 4
   *        then, start_index = 1, end_index = 5
   *
   *    if a value is not located contiguously, `KeyError` will be thrown
   *  ex: index = [..., x, x, ..., x, ...], value = x --> KeyError
   * @param start_value
   * @param end_value
   * @param start_index
   * @param end_index
   * @return
   */
  Status LocationRangeByValue(const std::shared_ptr<arrow::Scalar> &start_value,
                              const std::shared_ptr<arrow::Scalar> &end_value,
                              int64_t *start_index,
                              int64_t *end_index);

  /**
   * Recalculate index with a new array
   * @param index_arr
   */
  virtual Status SetIndexArray(std::shared_ptr<arrow::Array> index_arr, int col_id = -1) = 0;

  /**
   * Get index values as an array
   * @return
   */
  virtual const std::shared_ptr<arrow::Array> &GetIndexArray() = 0;

  virtual IndexingType GetIndexingType() = 0;

  /**
   * Checks if the index values are unique
   * @return
   */
  virtual bool IsUnique() = 0;

  int GetColId() const { return col_id_; }

  int64_t GetSize() const { return size_; }

  arrow::MemoryPool *GetPool() const { return pool_; }

  void SetColId(int col_id) { col_id_ = col_id; }

  /**
   * Slice the current index in the range [start, end), and return a new index
   * @param start
   * @param end
   * @return
   */
  virtual Status Slice(int64_t start, int64_t end, std::shared_ptr<BaseArrowIndex> *out_index) const = 0;

 private:
  int64_t size_;
  // col_id would be <0 if the index is set with an array other than a column (ex: range index)
  int col_id_;
  arrow::MemoryPool *pool_;
};

// ------------------ Range Index ------------------
/**
 * Builds range index from a column in the range [start, end)
 * @param input_table
 * @param start
 * @param end if -1, end is inferred to be the (input_table.num_rows())
 * @param step
 * @param pool
 * @return
 */
std::shared_ptr<BaseArrowIndex> BuildRangeIndex(const std::shared_ptr<arrow::Table> &input_table,
                                                int64_t start = 0,
                                                int64_t end = -1,
                                                int64_t step = 1,
                                                arrow::MemoryPool *pool = arrow::default_memory_pool());


// ------------------ Linear Index ------------------
/**
 * Builds linear index from a column
 * @param table
 * @param col_id
 * @param out_index
 * @param pool
 * @return
 */
Status BuildLinearIndex(const std::shared_ptr<arrow::Table> &table, int col_id,
                        std::shared_ptr<BaseArrowIndex> *out_index,
                        arrow::MemoryPool *pool = arrow::default_memory_pool());
Status BuildLinearIndex(std::shared_ptr<arrow::Array> index_array,
                        std::shared_ptr<BaseArrowIndex> *out_index,
                        int col_id = BaseArrowIndex::kNoColumnId,
                        arrow::MemoryPool *pool = arrow::default_memory_pool());


// ------------------ Hash Index ------------------
/**
 * Builds hash index from a column
 * @param table
 * @param col_id
 * @param output
 * @param pool
 * @return
 */
Status BuildHashIndex(const std::shared_ptr<arrow::Table> &table,
                      int col_id,
                      std::shared_ptr<BaseArrowIndex> *output,
                      arrow::MemoryPool *pool = arrow::default_memory_pool());
Status BuildHashIndex(std::shared_ptr<arrow::Array> index_array,
                      std::shared_ptr<BaseArrowIndex> *output,
                      int col_id = BaseArrowIndex::kNoColumnId,
                      arrow::MemoryPool *pool = arrow::default_memory_pool());

/**
 * Builds index for an arrow table
 * @param table
 * @param col_id
 * @param indexing_type
 * @param output
 * @param pool
 * @return
 */
Status BuildIndex(const std::shared_ptr<arrow::Table> &table,
                  int col_id,
                  IndexingType indexing_type,
                  std::shared_ptr<BaseArrowIndex> *output,
                  arrow::MemoryPool *pool = arrow::default_memory_pool());
Status BuildIndex(std::shared_ptr<arrow::Array> index_array,
                  IndexingType indexing_type,
                  std::shared_ptr<BaseArrowIndex> *output,
                  int col_id = BaseArrowIndex::kNoColumnId,
                  arrow::MemoryPool *pool = arrow::default_memory_pool());
/**
 * Sets index for a cylon table, and creates a new table
 * @param indexing_type
 * @param table
 * @param col_id
 * @param output
 * @return
 */
Status BuildIndexFromTable(const std::shared_ptr<Table> &table,
                           int col_id,
                           IndexingType indexing_type,
                           std::shared_ptr<Table> *out_table,
                           bool drop = false);

Status SetIndexForTable(std::shared_ptr<Table> &table,
                        std::shared_ptr<arrow::Array> array,
                        IndexingType indexing_type,
                        int col_id = BaseArrowIndex::kNoColumnId);

}  // namespace cylon

#endif  // CYLON_SRC_CYLON_INDEXING_INDEX_H_

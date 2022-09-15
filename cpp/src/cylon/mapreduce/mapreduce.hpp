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

#ifndef CYLON_CPP_SRC_CYLON_MAPREDUCE_MAPREDUCE_HPP_
#define CYLON_CPP_SRC_CYLON_MAPREDUCE_MAPREDUCE_HPP_

#include <arrow/api.h>
#include <utility>

#include "cylon/table.hpp"
#include "cylon/compute/aggregate_kernels.hpp"

namespace cylon {
namespace mapred {

struct MapToGroupKernel {
 public:
  explicit MapToGroupKernel(arrow::MemoryPool *pool = arrow::default_memory_pool()) : pool_(pool) {}

  virtual ~MapToGroupKernel() = default;

  /**
   * Map each row to a unique group id in the range [0, local_num_groups). This is a local
   * operation.
   * @param arrays
   * @param local_group_ids unique group id (length = num rows)
   * @param local_group_indices index of each unique group id (length = local_num_groups)
   * @param local_num_groups number of unique (local) groups
   * @return
   */
  virtual Status Map(const arrow::ArrayVector &arrays,
                     std::shared_ptr<arrow::Array> *local_group_ids,
                     std::shared_ptr<arrow::Array> *local_group_indices,
                     int64_t *local_num_groups) const;

  Status Map(const std::shared_ptr<arrow::Table> &table,
             const std::vector<int> &key_cols,
             std::shared_ptr<arrow::Array> *local_group_ids,
             std::shared_ptr<arrow::Array> *local_group_indices,
             int64_t *local_num_groups) const;

 private:
  arrow::MemoryPool *pool_;
};

/**
 * Reduce an array is a distributed fashion. It is done in the following stages.
 *  1. MapToGroups: Calculate group_ids for value_col
 *  2. CombineLocally: Combine value_col locally based on group_ids (which creates an intermediate array vector)
 *  3. Shuffle: Shuffle a temp table with intermediate results
 *  4. MapToGroups: Calculate group_ids for shuffled intermediate results
 *  5. ReduceShuffledResults: Reduce shuffled intermediate results (which creates a reduced array vector)
 *  6. Finalize: Finalize the reduced arrays
 *
 *  ex: take `mean` operation
 *  1. Calculate group_ids for value_col
 *  2. Locally calculate sum and count for each group_id (intermediate_arrays = {sums, cnts})
 *  3. Shuffle intermediate_array
 *  4. Calculate group_ids for shuffled intermediate results
 *  5. Reduce shuffled sums and counts individually (reduced_arrays = {reduced_sums, reduced_cnts})
 *  6. output = divide(reduced_sum/ reduced_cnts)
 *
 *  In a serial execution mode, this will be simplified into following stages.
 *  1. MapToGroups: Calculate group_ids for value_col
 *  2. CombineLocally: Combine value_col locally based on group_ids (which creates an intermediate array vector)
 *  3. Finalize: Finalize the intermediate arrays
 */
struct MapReduceKernel {
 public:
  virtual ~MapReduceKernel() = default;

  virtual void Init(arrow::MemoryPool *pool, compute::KernelOptions *options) = 0;

  /**
   * Combine `value_col` array locally based on the group_id, and push intermediate results to
   * `combined_results` array vector.
   * @param value_col
   * @param local_group_ids
   * @param local_num_groups
   * @param combined_results
   * @return
   */
  virtual Status CombineLocally(const std::shared_ptr<arrow::Array> &value_col,
                                const std::shared_ptr<arrow::Array> &local_group_ids,
                                int64_t local_num_groups,
                                arrow::ArrayVector *combined_results) const = 0;

  /**
   * Reduce `combined_results` vector to its finalized array vector based on the new group_ids
   * (after being shuffled).
   * @param combined_results
   * @param local_group_ids
   * @param local_group_indices
   * @param local_num_groups
   * @param reduced_results
   * @return
   */
  virtual Status ReduceShuffledResults(const arrow::ArrayVector &combined_results,
                                       const std::shared_ptr<arrow::Array> &local_group_ids,
                                       const std::shared_ptr<arrow::Array> &local_group_indices,
                                       int64_t local_num_groups,
                                       arrow::ArrayVector *reduced_results) const = 0;

  /**
   * Create the final output array
   * @param combined_results
   * @param output
   * @return
   */
  virtual Status Finalize(const arrow::ArrayVector &combined_results,
                          std::shared_ptr<arrow::Array> *output) const = 0;

  /**
   * In distributed mode, some kernel implementations may choose to do a single stage reduction of
   * an array. i.e.
   *    Shuffle --> MapToGroups --> CombineLocally --> Finalize
   * Those can simply set this flag. Then the shuffled value column will be forwarded straight to
   * the CombineLocally method.
   */
  virtual bool single_stage_reduction() const { return false; };
  inline size_t num_arrays() const;
  virtual std::string name() const = 0;
  virtual const std::shared_ptr<arrow::DataType> &output_type() const = 0;
  virtual const arrow::DataTypeVector &intermediate_types() const = 0;
};

std::unique_ptr<MapReduceKernel> MakeMapReduceKernel(const std::shared_ptr<arrow::DataType> &type,
                                                     compute::AggregationOpId reduce_op);

/**
 * Distributed hash groupby using mapreduce approach
 */
using AggOpVector = std::vector<std::pair<int, std::shared_ptr<compute::AggregationOp>>>;
Status MapredHashGroupBy(const std::shared_ptr<Table> &table, const std::vector<int> &key_cols,
                         const AggOpVector &aggs, std::shared_ptr<Table> *output,
                         const std::unique_ptr<MapToGroupKernel> &mapper
                         = std::make_unique<MapToGroupKernel>());

using AggOpIdVector = std::vector<std::pair<int, compute::AggregationOpId>>;
Status MapredHashGroupBy(const std::shared_ptr<Table> &table, const std::vector<int> &key_cols,
                         const AggOpIdVector &aggs, std::shared_ptr<Table> *output);

}
}

#endif //CYLON_CPP_SRC_CYLON_MAPREDUCE_MAPREDUCE_HPP_

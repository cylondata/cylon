//
// Created by niranda on 11/22/21.
//

#ifndef CYLON_CPP_SRC_CYLON_MAPREDUCE_MAPREDUCE_HPP_
#define CYLON_CPP_SRC_CYLON_MAPREDUCE_MAPREDUCE_HPP_

#include <arrow/api.h>
#include <utility>

#include "cylon/table.hpp"
#include "cylon/compute/aggregate_kernels.hpp"

namespace cylon {
namespace mapred {

struct MapToGroupKernel {
  virtual ~MapToGroupKernel() = default;
  virtual Status Map(const arrow::ArrayVector &arrays,
                     std::shared_ptr<arrow::Array> *local_group_ids,
                     std::shared_ptr<arrow::Array> *local_group_indices,
                     int64_t *local_num_groups,
                     arrow::MemoryPool *pool) const;

  Status Map(const std::shared_ptr<arrow::Table> &table,
             const std::vector<int> &key_cols,
             std::shared_ptr<arrow::Array> *local_group_ids,
             std::shared_ptr<arrow::Array> *local_group_indices,
             int64_t *local_num_groups,
             arrow::MemoryPool *pool = arrow::default_memory_pool()) const;
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
  virtual ~MapReduceKernel() = default;

  virtual void Init(arrow::MemoryPool *pool, compute::KernelOptions *options) = 0;

  virtual Status CombineLocally(const std::shared_ptr<arrow::Array> &value_col,
                                const std::shared_ptr<arrow::Array> &local_group_ids, int64_t local_num_groups,
                                arrow::ArrayVector *combined_results) const = 0;

  virtual Status ReduceShuffledResults(const arrow::ArrayVector &combined_results,
                                       const std::shared_ptr<arrow::Array> &local_group_ids,
                                       const std::shared_ptr<arrow::Array> &local_group_indices,
                                       int64_t local_num_groups,
                                       arrow::ArrayVector *reduced_results) const = 0;

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
 * 1. map keys to groups in the local table
 * 2. combine locally
 * 3. shuffle combined results
 * 4. map keys to groups in the shuffled table
 * 5. reduce shuffled table locally
 * 6. finalize reduction
 */
using AggOpVector = std::vector<std::pair<int, compute::AggregationOp *>>;

Status HashGroupByAggregate(const std::shared_ptr<Table> &table, const std::vector<int> &key_cols,
                            const AggOpVector &aggs, std::shared_ptr<Table> *output,
                            const std::unique_ptr<MapToGroupKernel> &mapper
                            = std::make_unique<MapToGroupKernel>());

}
}

#endif //CYLON_CPP_SRC_CYLON_MAPREDUCE_MAPREDUCE_HPP_

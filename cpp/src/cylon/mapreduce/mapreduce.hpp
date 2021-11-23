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
  virtual Status Map(const std::shared_ptr<CylonContext> &ctx,
                     const std::vector<std::shared_ptr<arrow::Array>> &arrays,
                     std::shared_ptr<arrow::Array> *local_group_ids,
                     std::shared_ptr<arrow::Array> *local_group_indices,
                     int64_t *local_num_groups) const;
};

/**
 * 1. map keys to groups in the local table
 * 2. combine locally
 * 3. shuffle combined results
 * 4. map keys to groups in the shuffled table
 * 5. reduce shuffled table locally
 * 6. finalize reduction
 */
struct MapReduceKernel {
  virtual Status Init(const std::shared_ptr<CylonContext> &ctx, int64_t local_num_groups,
                      arrow::ArrayVector *combined_results) const = 0;

  virtual Status Combine(const std::shared_ptr<CylonContext> &ctx,
                         const std::shared_ptr<arrow::Array> &value_col,
                         const int64_t *local_group_ids,
                         int64_t local_num_groups,
                         arrow::ArrayVector *combined_results) const = 0;

  virtual Status Reduce(const std::shared_ptr<CylonContext> &ctx,
                        const arrow::ArrayVector &combined_results,
                        const int64_t *local_group_ids,
                        int64_t local_num_groups,
                        arrow::ArrayVector *reduced_results) const = 0;

  virtual Status Finalize(const std::shared_ptr<CylonContext> &ctx,
                          const arrow::ArrayVector &reduced_results,
                          std::shared_ptr<arrow::Array> *output) const = 0;
};

struct CombineReduceConfig {
  int col_id;
  compute::AggregationOpId reduce_op;
};

Status MapReduce(const std::shared_ptr<Table> &table,
                 std::vector<int> key_cols,
                 std::vector<CombineReduceConfig> reduce_configs,
                 std::shared_ptr<Table> *output);

}
}

#endif //CYLON_CPP_SRC_CYLON_MAPREDUCE_MAPREDUCE_HPP_

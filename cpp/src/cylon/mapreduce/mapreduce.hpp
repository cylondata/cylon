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
  virtual Status Map(const std::shared_ptr<CylonContext> &ctx,
                     const arrow::ArrayVector &arrays,
                     std::shared_ptr<arrow::Array> *local_group_ids,
                     std::shared_ptr<arrow::Array> *local_group_indices,
                     int64_t *local_num_groups) const;

  Status Map(const std::shared_ptr<CylonContext> &ctx,
             const std::shared_ptr<arrow::Table> &table,
             const std::vector<int> &key_cols,
             std::shared_ptr<arrow::Array> *local_group_ids,
             std::shared_ptr<arrow::Array> *local_group_indices,
             int64_t *local_num_groups) const;
};

struct MapReduceKernel {
  virtual ~MapReduceKernel() = default;

  virtual Status Init(const std::shared_ptr<CylonContext> &ctx, int64_t local_num_groups,
                      arrow::ArrayVector *combined_results) = 0;

  virtual Status Combine(const std::shared_ptr<arrow::Array> &value_col,
                         const int64_t *local_group_ids, int64_t local_num_groups,
                         arrow::ArrayVector *combined_results) const = 0;

  virtual Status Reduce(const arrow::ArrayVector &combined_results, const int64_t *local_group_ids,
                        const int64_t *local_group_indices, int64_t local_num_groups,
                        arrow::ArrayVector *reduced_results) const = 0;

  virtual Status Finalize(const arrow::ArrayVector &reduced_results,
                          std::shared_ptr<arrow::Array> *output) const = 0;

  virtual size_t num_arrays() const = 0;
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
Status Aggregate(const std::shared_ptr<CylonContext> &ctx,
                 const std::shared_ptr<Table> &table,
                 const std::vector<int> &key_cols,
                 const std::vector<std::pair<int, compute::AggregationOpId>> &aggs,
                 std::shared_ptr<Table> *output,
                 const MapToGroupKernel &mapper = MapToGroupKernel());

}
}

#endif //CYLON_CPP_SRC_CYLON_MAPREDUCE_MAPREDUCE_HPP_

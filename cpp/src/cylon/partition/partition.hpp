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

#ifndef CYLON_CPP_SRC_CYLON_PARTITION_PARTITION_HPP_
#define CYLON_CPP_SRC_CYLON_PARTITION_PARTITION_HPP_

#include <status.hpp>
#include <memory>
#include <ctx/cylon_context.hpp>
#include <table.hpp>
namespace cylon {

/**
 * Builds target partitions based on the hash_column_idx
 * @param table
 * @param num_partitions
 * @param target_partitions
 * @param partition_histogram
 * @return
 */

Status PartitionByHashing(const std::shared_ptr<Table> &table,
                          int32_t hash_column_idx,
                          uint32_t num_partitions,
                          std::vector<uint32_t> &target_partitions,
                          std::vector<uint32_t> &partition_histogram);

Status PartitionByHashing(const std::shared_ptr<Table> &table,
                          const std::vector<int32_t> &hash_column_idx,
                          uint32_t num_partitions,
                          std::vector<uint32_t> &target_partitions,
                          std::vector<uint32_t> &partition_histogram);

/**
 * Sorted partitioning of the distributed table
 * @param table
 * @param idx_col
 * @param num_partitions
 * @param target_partitions
 * @param partition_histogram
 * @param ascending (optional) ascending/ descending
 * @param num_samples (optional) number of samples
 * @param num_bins (optional) number of bins
 * @return
 */
Status PartitionBySorting(const std::shared_ptr<Table> &table,
                       int32_t column_idx,
                          uint32_t num_partitions,
                          std::vector<uint32_t> &target_partitions,
                          std::vector<uint32_t> &partition_histogram,
                          bool ascending,
                          uint64_t num_samples,
                          uint32_t num_bins);

/**
 * split a table based on the @param target_partitions vector. target_partition elements [0, num_partitions).
 * Optionally provide a histogram of partitions, i.e. number of rows belonging for each target_partition.
 * Length of partition_histogram should be equal to num_partitions.
 * @param table
 * @param target_partitions
 * @param num_partitions
 * @param output
 * @param partition_hist_ptr
 * @return
 */
Status Split(const std::shared_ptr<Table> &table,
             uint32_t num_partitions,
             const std::vector<uint32_t> &target_partitions,
             std::vector<std::shared_ptr<arrow::Table>> &output);

Status Split(const std::shared_ptr<Table> &table,
             uint32_t num_partitions,
             const std::vector<uint32_t> &target_partitions,
             const std::vector<uint32_t> &partition_hist_ptr,
             std::vector<std::shared_ptr<arrow::Table>> &output);

//struct PartitionSplitter {
//
//  PartitionSplitter(const std::shared_ptr<arrow::Schema> &table_schema,
//                    const std::vector<int32_t> &col_indices,
//                    int32_t num_partitions)
//      : table_schema(table_schema), col_indices(col_indices), num_partitions(num_partitions) {
//
//  }
//
//  virtual Status Partition(std::vector<int32_t> &target_partitions,
//                           std::vector<uint32_t> &partition_hist) = 0;
//
//  virtual Status Split(std::vector<std::shared_ptr<Table>> &output,
//                       const std::vector<uint32_t> *partition_hist_ptr) = 0;
//
//  const std::shared_ptr<arrow::Schema> &table_schema;
//  const std::vector<int32_t> &col_indices;
//  const int32_t num_partitions;
//  std::vector<cylon::ArrowArraySplitKernel> split_kernels;
//};
//
//template<typename T>
//struct Histogram {
//  uint32_t num_bins;
//  std::vector<T> bin_boundaries; // lower bound of boundaries
//  std::vector<float_t> frequencies; // frequencies can be decimals
//};



}

#endif //CYLON_CPP_SRC_CYLON_PARTITION_PARTITION_HPP_

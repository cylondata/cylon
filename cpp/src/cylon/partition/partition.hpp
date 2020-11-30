//
// Created by niranda on 11/19/20.
//

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

Status ApplyPartition(const std::shared_ptr<Table> &table,
                      const std::vector<int32_t> &hash_column_idx,
                      uint32_t num_partitions,
                      std::vector<uint32_t> &target_partitions,
                      std::vector<uint32_t> &partition_histogram);

Status HashPartition(const std::shared_ptr<Table> &table,
                     int32_t hash_column_idx,
                     uint32_t num_partitions,
                     std::vector<uint32_t> &target_partitions,
                     std::vector<uint32_t> &partition_histogram);

Status HashPartition(const std::shared_ptr<Table> &table,
                     const std::vector<int32_t> &hash_column_idx,
                     uint32_t num_partitions,
                     std::vector<uint32_t> &target_partitions,
                     std::vector<uint32_t> &partition_histogram);

Status ModuloPartition(const std::shared_ptr<Table> &table,
                       const std::vector<int32_t> &hash_column_idx,
                       int32_t num_partitions,
                       std::vector<int32_t> &target_partitions,
                       std::vector<uint32_t> &partition_histogram);

Status ModuloPartition(const std::shared_ptr<Table> &table,
                       int32_t hash_column_idx,
                       uint32_t num_partitions,
                       std::vector<uint32_t> &target_partitions,
                       std::vector<uint32_t> &partition_hist);

Status RangePartition(const std::shared_ptr<Table> &table,
                      const std::vector<int32_t> &hash_column_idx,
                      uint32_t num_partitions,
                      std::vector<uint32_t> &target_partitions,
                      std::vector<uint32_t> &partition_histogram,
                      uint32_t num_samples = 0,
                      uint32_t num_bins = 0);

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
             const std::vector<uint32_t> &target_partitions,
             uint32_t num_partitions,
             std::vector<std::shared_ptr<Table>> &output,
             const std::vector<uint32_t> *partition_hist_ptr = nullptr);

struct PartitionSplitter {

  PartitionSplitter(const std::shared_ptr<arrow::Schema> &table_schema,
                    const std::vector<int32_t> &col_indices,
                    int32_t num_partitions)
      : table_schema(table_schema), col_indices(col_indices), num_partitions(num_partitions) {

  }

  virtual Status Partition(std::vector<int32_t> &target_partitions,
                           std::vector<uint32_t> &partition_hist) = 0;

  virtual Status Split(std::vector<std::shared_ptr<Table>> &output,
                       const std::vector<uint32_t> *partition_hist_ptr) = 0;

  const std::shared_ptr<arrow::Schema> &table_schema;
  const std::vector<int32_t> &col_indices;
  const int32_t num_partitions;
  std::vector<cylon::ArrowArraySplitKernel> split_kernels;
};

template<typename T>
struct Histogram {
  uint32_t num_bins;
  std::vector<T> bin_boundaries; // lower bound of boundaries
  std::vector<float_t> frequencies; // frequencies can be decimals
};



}

#endif //CYLON_CPP_SRC_CYLON_PARTITION_PARTITION_HPP_

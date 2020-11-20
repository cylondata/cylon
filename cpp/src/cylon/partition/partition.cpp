//
// Created by niranda on 11/19/20.
//

#include <glog/logging.h>
#include <ctx/arrow_memory_pool_utils.hpp>

#include "partition.hpp"
#include "util/macros.hpp"

namespace cylon {

static Status split_impl(const std::shared_ptr<Table> &table,
                         const std::vector<int32_t> &target_partitions,
                         int32_t num_partitions,
                         std::vector<std::shared_ptr<Table>> &output,
                         const std::vector<uint32_t> *partition_hist_ptr) {
  auto t1 = std::chrono::high_resolution_clock::now();
  Status status;
  const std::shared_ptr<arrow::Table> &arrow_table = table->get_table();
  std::shared_ptr<cylon::CylonContext> ctx = table->GetContext();
  arrow::MemoryPool *pool = cylon::ToArrowPool(ctx);

  std::vector<arrow::ArrayVector> data_arrays(num_partitions); // size num_partitions

  for (const auto &col:arrow_table->columns()) {
    std::shared_ptr<ArrowArraySplitKernel> splitKernel;
    status = CreateSplitter(col->type(), pool, &splitKernel);
    RETURN_IF_STATUS_FAILED(status)

    std::vector<std::shared_ptr<arrow::Array>> split_arrays;
    status = splitKernel->Split(col, target_partitions, num_partitions, *partition_hist_ptr, split_arrays);
    RETURN_IF_STATUS_FAILED(status)

    for (size_t i = 0; i < split_arrays.size(); i++) {
      data_arrays[i].push_back(split_arrays[i]);
    }
  }

  output.reserve(num_partitions);
  for (const auto &arr_vec: data_arrays) {
    std::shared_ptr<arrow::Table> arrow_table_out = arrow::Table::Make(arrow_table->schema(), arr_vec);
    std::shared_ptr<Table> cylon_table_out;
    status = cylon::Table::FromArrowTable(ctx, arrow_table_out, cylon_table_out);
    RETURN_IF_STATUS_FAILED(status)
    output.push_back(std::move(cylon_table_out));
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Splitting table time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  return Status::OK();
}

Status Split(const std::shared_ptr<Table> &table,
             const std::vector<int32_t> &target_partitions,
             int32_t num_partitions,
             std::vector<std::shared_ptr<Table>> &output,
             const std::vector<uint32_t> *partition_hist_ptr) {

  if ((size_t) table->Rows() != target_partitions.size()) {
    LOG_AND_RETURN_ERROR(Code::ExecutionError, "tables rows != target_partitions length")
  }

  if (partition_hist_ptr == nullptr) {
    LOG(INFO) << "building partition histogram";
    std::vector<uint32_t> partition_hist(num_partitions, 0);
    for (const int32_t p:target_partitions) {
      partition_hist[p]++;
    }

    return split_impl(table, target_partitions, num_partitions, output, &partition_hist);
  } else {
    return split_impl(table, target_partitions, num_partitions, output, partition_hist_ptr);
  }
}

}

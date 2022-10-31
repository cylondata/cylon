//
// Created by Mills "Bud" Staylor on 10/12/22.
//


#include "hash_window.hpp"

#include "cylon/thridparty/flat_hash_map/bytell_hash_map.hpp"
#include <arrow/api.h>
#include <arrow/visitor_inline.h>
#include <arrow/compute/api.h>
#include <chrono>
#include <glog/logging.h>

#include "cylon/arrow/arrow_comparator.hpp"
#include "cylon/ctx/arrow_memory_pool_utils.hpp"
#include "cylon/util/macros.hpp"

namespace cylon {
namespace windowing {

Status HashWindow(const config::WindowConfig &window_config,
                  const std::shared_ptr<Table> &table,
                  const std::vector<int32_t> &idx_cols,
                  const std::vector<std::pair<int32_t, compute::AggregationOpId>> &aggregate_cols,
                  std::shared_ptr<Table> &output) {
  std::vector<std::pair<int32_t, std::shared_ptr<compute::AggregationOp>>> aggregations;
  aggregations.reserve(aggregate_cols.size());
  for (auto &&p:aggregate_cols) {
    // create AggregationOp with nullptr options
    aggregations.emplace_back(p.first, compute::MakeAggregationOpFromID(p.second));
  }

  return HashWindow(window_config, table, idx_cols, aggregations, output);



}


/**
 * Hash group-by operation by using <col_index, AggregationOp> pairs
 * NOTE: Nulls in the value columns will be ignored!
 * @param table
 * @param idx_cols
 * @param aggregations
 * @param output
 * @return
 */
Status HashWindow(const config::WindowConfig &window_config,
                  const std::shared_ptr<Table> &table,
                  const std::vector<int32_t> &idx_cols,
                  const std::vector<std::pair<int32_t, std::shared_ptr<compute::AggregationOp>>> &aggregations,
                  std::shared_ptr<Table> &output) {

#ifdef CYLON_DEBUG
  auto t1 = std::chrono::steady_clock::now();
#endif
  const auto &ctx = table->GetContext();
  arrow::MemoryPool *pool = ToArrowPool(ctx);

  std::shared_ptr<arrow::Table> atable = table->get_table();

  //slide tables

  std::vector<Table> offsets;

  if (window_config.GetStep() > 0) {
    SlicesByObservations(window_config, table, offsets);
  }



  return cylon::Status();
}

Status SlicesByObservations(const config::WindowConfig &window_config,
                            const std::shared_ptr<Table> &table,
                            std::vector<Table> &output) {

}

Status SlicesByOffset(const config::WindowConfig &window_config,
                      const std::shared_ptr<Table> &table,
                      std::vector<Table> &output) {

}

Status HashWindow(const config::WindowConfig &window_config,
                  std::shared_ptr<Table> &table,
                  int32_t idx_col,
                  const std::vector<int32_t> &aggregate_cols,
                  const std::vector<compute::AggregationOpId> &aggregate_ops,
                  std::shared_ptr<Table> &output) {

  return cylon::Status();
}

}
}

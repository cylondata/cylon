//
// Created by Mills "Bud" Staylor on 10/12/22.
//

#include "window.hpp"

namespace cylon {
namespace windowing {

Status DistributedHashWindow(const config::WindowConfig &window_config,
                             std::shared_ptr<Table> &table,
                             const std::vector<int32_t> &index_cols,
                             const std::vector<int32_t> &aggregate_cols,
                             const std::vector<compute::AggregationOpId> &aggregate_ops,
                             std::shared_ptr<Table> &output) {






  return cylon::Status();
}
Status DistributedHashWindow(const config::WindowConfig &window_config,
                                                              std::shared_ptr<Table> &table,
                                                              int32_t index_col,
                                                              const std::vector<int32_t> &aggregate_cols,
                                                              const std::vector<compute::AggregationOpId> &aggregate_ops,
                                                              std::shared_ptr<Table> &output) {
  return cylon::Status();
}
}
}

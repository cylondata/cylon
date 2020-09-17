//
// Created by niranda on 9/17/20.
//

#ifndef CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_HPP_
#define CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_HPP_

#include <status.hpp>
#include <table.hpp>
#include <compute/aggregates.hpp>

namespace cylon {

Status GroupBy(const std::shared_ptr<Table> &table,
               int64_t index_col,
               const std::vector<int64_t> &aggregate_cols,
               const std::vector<cylon::compute::AggregateOperation> &aggregate_ops,
               std::shared_ptr<Table> &output);

}

#endif //CYLON_CPP_SRC_CYLON_GROUPBY_GROUPBY_HPP_

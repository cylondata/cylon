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

#include "../util/arrow_utils.hpp"
#include "../util/macros.hpp"

#include "hash_groupby.hpp"
#include "pipeline_groupby.hpp"
#include "groupby.hpp"

namespace cylon {

static const std::vector<compute::AggregationOpId>
    ASSOCIATIVE_OPS{compute::SUM, compute::MIN, compute::MAX};

static inline bool is_associative(const std::vector<compute::AggregationOpId> &aggregate_ops) {
  return std::all_of(aggregate_ops.begin(), aggregate_ops.end(), [](const compute::AggregationOpId &op) {
    return std::find(ASSOCIATIVE_OPS.begin(), ASSOCIATIVE_OPS.end(), op) != ASSOCIATIVE_OPS.end();
  });
}

Status DistributedHashGroupBy(std::shared_ptr<Table> &table,
                              const std::vector<int32_t> &index_cols,
                              const std::vector<int32_t> &aggregate_cols,
                              const std::vector<compute::AggregationOpId> &aggregate_ops,
                              std::shared_ptr<Table> &output) {
  if (aggregate_cols.size() != aggregate_ops.size()) {
    return Status(Code::Invalid, "aggregate_cols size != aggregate_ops size");
  }

  // first filter index + aggregation cols
  std::vector<int32_t> project_cols(index_cols);
  project_cols.insert(project_cols.end(), aggregate_cols.begin(), aggregate_cols.end());
  std::shared_ptr<Table> projected_table;
  LOG_AND_RETURN_CYLON_STATUS_IF_FAILED(Project(table, project_cols, projected_table));

  // adjust local column indices for aggregations
  std::vector<int32_t> indices_after_project(index_cols.size());
  std::iota(indices_after_project.begin(), indices_after_project.end(), 0);

  std::vector<std::pair<int32_t, compute::AggregationOpId>> agg_after_projection;
  for (size_t i = 0; i < aggregate_cols.size(); i++) {
    agg_after_projection.emplace_back(index_cols.size() + i, aggregate_ops[i]);
  }

  // do local group by if world_sz is 1 or if all agg ops are associative
  // todo: find a better way to do this, rather than checking the associativity
  std::shared_ptr<Table> local_table;
  if (table->GetContext()->GetWorldSize() == 1 || is_associative(aggregate_ops)) {
    LOG_AND_RETURN_CYLON_STATUS_IF_FAILED(HashGroupBy(projected_table,
                                                      indices_after_project,
                                                      agg_after_projection,
                                                      local_table));
  } else {
    local_table = std::move(projected_table);
  }

  if (table->GetContext()->GetWorldSize() > 1) {
    // shuffle
    LOG_AND_RETURN_CYLON_STATUS_IF_FAILED(Shuffle(local_table, indices_after_project, local_table));

    // do local distribute again
    LOG_AND_RETURN_CYLON_STATUS_IF_FAILED(HashGroupBy(local_table,
                                                      indices_after_project,
                                                      agg_after_projection,
                                                      output));
  } else {
    output = local_table;
  }

  return Status::OK();
}

Status DistributedHashGroupBy(std::shared_ptr<Table> &table,
                              int32_t index_col,
                              const std::vector<int32_t> &aggregate_cols,
                              const std::vector<compute::AggregationOpId> &aggregate_ops,
                              std::shared_ptr<Table> &output) {
  return DistributedHashGroupBy(table, std::vector<int32_t>{index_col}, aggregate_cols, aggregate_ops, output);
}

Status DistributedPipelineGroupBy(std::shared_ptr<Table> &table,
                                  int32_t index_col,
                                  const std::vector<int32_t> &aggregate_cols,
                                  const std::vector<compute::AggregationOpId> &aggregate_ops,
                                  std::shared_ptr<Table> &output) {

  if (aggregate_cols.size() != aggregate_ops.size()) {
    return Status(Code::Invalid, "aggregate_cols size != aggregate_ops size");
  }

  // first filter index + aggregation cols
  std::vector<int32_t> project_cols{index_col};
  project_cols.insert(project_cols.end(), aggregate_cols.begin(), aggregate_cols.end());
  std::shared_ptr<Table> projected_table;
  LOG_AND_RETURN_CYLON_STATUS_IF_FAILED(Project(table, project_cols, projected_table));

  // adjust local column indices for aggregations
  std::vector<std::pair<int32_t, compute::AggregationOpId>> agg_after_projection;
  for (size_t i = 0; i < aggregate_cols.size(); i++) {
    agg_after_projection.emplace_back(1 + i, aggregate_ops[i]);
  }

  // do local group by
  std::shared_ptr<Table> local_table;
  // todo: find a better way to do this, rather than checking the associativity
  if (table->GetContext()->GetWorldSize() == 1 || is_associative(aggregate_ops)) {
    LOG_AND_RETURN_CYLON_STATUS_IF_FAILED(PipelineGroupBy(projected_table, 0, agg_after_projection, local_table));
  } else {
    local_table = std::move(projected_table);
  }

  if (table->GetContext()->GetWorldSize() > 1) {
    // shuffle
    LOG_AND_RETURN_CYLON_STATUS_IF_FAILED(Shuffle(local_table, {0}, local_table));

    // sort the table after the shuffle
    LOG_AND_RETURN_CYLON_STATUS_IF_FAILED(Sort(local_table, 0, local_table));

    // do local distribute again.
    LOG_AND_RETURN_CYLON_STATUS_IF_FAILED(PipelineGroupBy(local_table, 0, agg_after_projection, output));
  } else {
    output = local_table;
  }
  return Status::OK();
}

}
